from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import ProviderConfig, build_chat_model, normalize_provider


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path) -> LabConfig:
    provider = ProviderConfig("openai", "gpt-4o-mini", 0)
    return LabConfig(REPO_ROOT, REPO_ROOT / "data", tmp_path / "state", 200, 2, provider, provider, offline=True)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("alice") == ""
    assert store.file_size("alice") == 0
    original = "# User\n- location: Huế\n- note: Huế\n"
    path = store.write_text("alice", original)
    assert path == tmp_path / "profiles" / "alice" / "User.md"
    assert store.read_text("alice") == original
    assert store.file_size("alice") == len(original.encode("utf-8"))
    assert store.edit_text("alice", "Huế", "Đà Nẵng")
    assert store.read_text("alice") == original.replace("Huế", "Đà Nẵng", 1)
    assert not store.edit_text("alice", "missing", "new")
    assert not store.edit_text("alice", "", "new")
    assert not store.edit_text("alice", "Huế", "Huế")
    assert UserProfileStore(tmp_path / "profiles").read_text("alice") == store.read_text("alice")
    assert store.read_text("bob") == ""


def test_profile_paths_are_isolated(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    users = ["alice", "../alice", "a/b", "a_b", "Nguyễn An"]
    paths = [store.write_text(user, user) for user in users]
    assert len(set(paths)) == len(users)
    for user, path in zip(users, paths):
        assert path.is_relative_to(tmp_path / "profiles")
        assert store.read_text(user) == user
    with pytest.raises(ValueError):
        store.path_for("")


def test_profile_upserts_do_not_duplicate_facts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.update_facts("alice", {"location": "Huế", "interests": "Python, AI"})
    store.upsert_fact("alice", "location", "Đà Nẵng")
    store.update_facts("alice", {"interests": "Python, RAG"})
    content = store.read_text("alice")
    size = store.file_size("alice")
    for _ in range(10):
        store.update_facts("alice", {"location": "Đà Nẵng", "interests": "Python, RAG"})
    assert store.file_size("alice") == size
    assert content.count("- location:") == 1
    assert "Huế" not in content
    assert store.facts("alice")["interests"] == "Python, AI, RAG"


@pytest.mark.parametrize(("text", "expected"), [("", 0), ("   ", 0), ("abcd", 1), ("abcde", 2), (" Huế ", 1)])
def test_token_estimator(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("Mình tên là Linh. Mình ở Hải Phòng và đang làm data engineer.", {"name": "Linh", "location": "Hải Phòng", "profession": "data engineer"}),
    ("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.", {"profession": "MLOps engineer"}),
    ("Giờ mình đang ở Huế chứ không còn ở Đà Nẵng.", {"location": "Huế"}),
    ("Mình muốn bạn trả lời ngắn gọn thành 3 bullet có ví dụ thực chiến.", {"response_style": "ngắn gọn; 3 bullet; có ví dụ thực chiến"}),
    ("Mình tên là An. Mình tên gì?", {"name": "An"}),
])
def test_extract_explicit_facts_and_corrections(text: str, expected: dict[str, str]) -> None:
    updates = extract_profile_updates(text)
    for key, value in expected.items():
        assert updates[key] == value


@pytest.mark.parametrize("text", [
    "Mình tên gì?", "Hiện tại mình đang ở đâu?", "Nếu mình ở Hà Nội thì sao?",
    "Nếu sau này mình làm product manager, bạn thấy sao.",
    "Có lúc mình đùa rằng hay là chuyển sang product manager cho vui, nhưng đó chỉ là câu đùa.",
    "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.",
    "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé, vì đó là thông tin cũ.",
    "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
    "Nhắc lại giúp mình tên và style trả lời mình thích trong stress test này.",
    "Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.",
])
def test_questions_and_noise_are_not_profile_updates(text: str) -> None:
    assert extract_profile_updates(text) == {}


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    manager.append("long", "user", "Mình tên là Linh. " + "Thông tin chi tiết. " * 60)
    manager.append("long", "assistant", "Đã ghi nhận.")
    manager.append("long", "user", "Nội dung mới.")
    context = manager.context("long")
    assert manager.compaction_count("long") == 1
    assert len(context["messages"]) == 2
    assert "Linh" in context["summary"]
    assert context["messages"][-1]["content"] == "Nội dung mới."
    assert manager.compaction_count("other") == 0
    assert manager.context("other")["messages"] == []


def test_repeated_compaction_preserves_prior_summary() -> None:
    manager = CompactMemoryManager(200, 2)
    manager.append("thread", "user", "Mình tên là Linh. " + "Ghi chú đầu tiên. " * 80)
    for index in range(10):
        manager.append("thread", "user", f"Phần tiếp {index}. " + "Nội dung dài. " * 80)
        manager.append("thread", "assistant", "Đã ghi nhận.")
    context = manager.context("thread")
    assert context["compactions"] > 1
    assert "Linh" in context["summary"]
    assert len(context["summary"]) <= 400
    assert len(context["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced, baseline = AdvancedAgent(config, force_offline=True), BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply("alice", "first", "Mình tên là Linh. Đồ uống yêu thích là trà đào.")
        assert "Linh" in agent.reply("alice", "first", "Mình tên gì?")["answer"]
    query = "Mình tên gì và đồ uống yêu thích là gì?"
    assert recall_points(advanced.reply("alice", "new", query)["answer"], ["Linh", "trà đào"]) == 1
    assert recall_points(baseline.reply("alice", "new", query)["answer"], ["Linh", "trà đào"]) == 0
    restarted = AdvancedAgent(config, force_offline=True)
    assert "Linh" in restarted.reply("alice", "after-restart", "Mình tên gì?")["answer"]
    assert "Linh" not in restarted.reply("bob", "bob-thread", "Mình tên gì?")["answer"]


def test_correction_survives_noise_and_restart(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    turns = ["Mình ở Huế và đang làm backend engineer.",
             "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
             "Giờ mình đang ở Đà Nẵng chứ không còn ở Huế.",
             "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.",
             "Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
             "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé, vì đó là thông tin cũ."]
    for turn in turns:
        agent.reply("alice", "thread", turn)
    restarted = AdvancedAgent(config, force_offline=True)
    answer = restarted.reply("alice", "new", "Nhắc lại nghề nghiệp và nơi ở hiện tại.")["answer"]
    assert "MLOps engineer" in answer and "Đà Nẵng" in answer
    assert "backend engineer" not in answer and "Huế" not in answer
    assert "product manager" not in answer and "Hà Nội" not in answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced, baseline = AdvancedAgent(config, force_offline=True), BaselineAgent(config, force_offline=True)
    turns = ["Mình tên là Linh."] + [f"Đoạn {index}. " + "Báo cáo kỹ thuật dài về các lựa chọn vận hành. " * 80 for index in range(15)]
    for message in turns:
        for agent in (advanced, baseline):
            agent.reply("alice", "long", message)
    assert advanced.compaction_count("long") > 1
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.6
    assert "Linh" in advanced.reply("alice", "new", "Mình tên gì?")["answer"]


@pytest.mark.parametrize("agent_type", [BaselineAgent, AdvancedAgent])
def test_accounting_and_thread_ownership(tmp_path: Path, agent_type) -> None:
    agent = agent_type(make_config(tmp_path), force_offline=True)
    first = agent.reply("alice", "one", "Mình tên là Linh.")
    second = agent.reply("alice", "one", "Mình tên gì?")
    assert agent.token_usage("one") == first["agent_tokens"] + second["agent_tokens"]
    assert agent.prompt_token_usage("one") == first["prompt_tokens"] + second["prompt_tokens"]
    assert agent.token_usage("missing") == agent.prompt_token_usage("missing") == 0
    with pytest.raises(ValueError):
        agent.reply("bob", "one", "Mình tên gì?")


def test_recall_and_quality_rules() -> None:
    assert recall_points("Không biết.", ["Linh", "trà đào"]) == 0
    assert recall_points("Linh", ["Linh", "trà đào"]) == 0.5
    assert recall_points("LINH thích TRÀ ĐÀO.", ["Linh", "trà đào"]) == 1
    assert recall_points("anything", []) == 0
    assert heuristic_quality("Linh", ["Linh"]) > heuristic_quality("Linh " * 500, ["Linh"])
    assert heuristic_quality("Không biết.", ["Linh"]) == 0


@pytest.mark.parametrize("filename", ["conversations.json", "advanced_long_context.json"])
def test_real_datasets(tmp_path: Path, filename: str) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1000, compact_keep_messages=4)
    conversations = load_conversations(config.data_dir / filename)
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    advanced_agent = AdvancedAgent(config, force_offline=True)
    advanced = run_agent_benchmark("Advanced", advanced_agent, conversations, config)
    assert baseline.recall_score == 0
    assert advanced.recall_score == 1
    assert baseline.memory_growth_bytes == baseline.compactions == 0
    assert advanced.memory_growth_bytes > 0
    if filename == "advanced_long_context.json":
        assert advanced.compactions > 1
        assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed
        facts = advanced_agent.profile_store.facts("dungct_stress")
        assert facts["location"] == "Đà Nẵng"
        assert facts["profession"] == "MLOps engineer"
        assert "3 bullet" in facts["response_style"]


def test_benchmark_recall_timing_and_accounting(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    conversations = [
        {"id": "a", "user_id": "alice", "turns": ["Mình ở Huế."],
         "recall_questions": [{"question": "Hiện tại mình đang ở đâu?", "expected_contains": ["Huế"]}]},
        {"id": "b", "user_id": "alice", "turns": ["Giờ mình đang ở Đà Nẵng."],
         "recall_questions": [{"question": "Hiện tại mình đang ở đâu?", "expected_contains": ["Đà Nẵng"]}]},
    ]
    agent = AdvancedAgent(config, force_offline=True)
    row = run_agent_benchmark("Advanced", agent, conversations, config)
    assert row.recall_score == 1
    assert row.agent_tokens_only == sum(agent.thread_tokens.values())
    assert row.prompt_tokens_processed == sum(agent.thread_prompt_tokens.values())
    assert row.memory_growth_bytes == agent.memory_file_size("alice")


def test_config_defaults_env_and_validation(tmp_path: Path, monkeypatch) -> None:
    for key in list(__import__("os").environ):
        if key.startswith(("LLM_", "JUDGE_", "COMPACT_")) or key in {"LAB_OFFLINE", "STATE_DIR"}:
            monkeypatch.delenv(key)
    (tmp_path / ".env").write_text("LLM_PROVIDER=anthorpic\nLLM_MODEL=from-file\nCOMPACT_THRESHOLD_TOKENS=333\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-environment")
    config = load_config(tmp_path)
    assert config.model.provider == "anthropic"
    assert config.model.model_name == "from-environment"
    assert config.judge_model.model_name == "from-environment"
    assert config.offline and config.state_dir.exists()
    assert config.compact_threshold_tokens == 333
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "0")
    with pytest.raises(ValueError):
        load_config(tmp_path)


@pytest.mark.parametrize("provider", ["openai", "custom", "gemini", "anthropic", "ollama", "openrouter"])
def test_provider_factory_routing(provider: str, monkeypatch) -> None:
    import model_provider
    calls = []

    class StubModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_import(module):
        from types import SimpleNamespace
        calls.append(module)
        return SimpleNamespace(**{name: StubModel for name in ("ChatOpenAI", "ChatGoogleGenerativeAI", "ChatAnthropic", "ChatOllama", "ChatOpenRouter")})

    monkeypatch.setattr(model_provider, "import_module", fake_import)
    config = ProviderConfig(provider, "test-model", 0.2, "test-key", "http://localhost:9999")
    model = build_chat_model(config)
    assert len(calls) == 1
    assert model.kwargs["model"] == "test-model"
    assert model.kwargs["temperature"] == 0.2
    if provider == "custom":
        assert model.kwargs["base_url"] == config.base_url
    if provider == "openrouter":
        assert model.kwargs["openrouter_api_key"] == "test-key"


def test_invalid_provider_and_missing_live_credentials() -> None:
    assert normalize_provider(" Anthorpic ") == "anthropic"
    with pytest.raises(ValueError):
        normalize_provider("invalid")
    with pytest.raises(ValueError):
        build_chat_model(ProviderConfig("openai", "test", 0))
    with pytest.raises(ValueError):
        build_chat_model(ProviderConfig("custom", "test", 0))


def test_invalid_dataset(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"id": "wrong-shape"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_conversations(path)


def test_duplicate_field_correction(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.write_text("alice", "# User\n- location: Huế\n- location: Hà Nội\n")
    store.upsert_fact("alice", "location", "Đà Nẵng")
    assert store.facts("alice")["location"] == "Đà Nẵng"
    assert store.read_text("alice").count("- location:") == 1
    assert "Huế" not in store.read_text("alice")


def test_style_updates_preserve_unspecified_dimensions(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.update_facts("alice", {"response_style": "ngắn gọn; 3 bullet; có ví dụ thực tế"})
    store.update_facts("alice", {"response_style": "có ví dụ thực chiến"})
    style = store.facts("alice")["response_style"]
    assert "ngắn gọn" in style and "3 bullet" in style and "ví dụ thực chiến" in style
    assert "ví dụ thực tế" not in style
    store.update_facts("alice", {"response_style": "chi tiết; 5 bullet"})
    style = store.facts("alice")["response_style"]
    assert "ngắn gọn" not in style and "3 bullet" not in style
    assert "chi tiết" in style and "5 bullet" in style


def test_offline_response_reads_summary_and_prioritizes_profile(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    context = agent.compact_memory.context("thread")
    context["summary"] = "# Summary\n- name: Linh\n- location: Huế"
    assert "Linh" in agent._offline_response("alice", "thread", "Mình tên gì?")
    agent.profile_store.upsert_fact("alice", "location", "Đà Nẵng")
    answer = agent._offline_response("alice", "thread", "Hiện tại mình đang ở đâu?")
    assert "Đà Nẵng" in answer and "Huế" not in answer


def _fake_live_model(responses):
    fake_module = pytest.importorskip("langchain_core.language_models.fake_chat_models")
    from pydantic import Field

    class RecordingModel(fake_module.FakeMessagesListChatModel):
        seen: list = Field(default_factory=list)

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.seen.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    return RecordingModel(responses=responses)


@pytest.mark.parametrize("agent_type", [BaselineAgent, AdvancedAgent])
def test_live_runtime_context_compaction_and_usage(tmp_path: Path, monkeypatch, agent_type) -> None:
    pytest.importorskip("langchain")
    from langchain_core.messages import AIMessage
    from agent_baseline import message_text
    import agent_advanced
    import agent_baseline

    model = _fake_live_model([AIMessage(content="Đã ghi nhận.")])
    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: model)
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: model)
    config = replace(make_config(tmp_path), offline=False)
    agent = agent_type(config)
    replies = [agent.reply("alice", "one", "Mình tên là Linh. Mình ở Huế.")]
    for index in range(8):
        replies.append(agent.reply("alice", "one", f"Đoạn {index}. " + "Báo cáo vận hành. " * 80))
    assert all(reply["mode"] == "live" for reply in replies)
    assert agent.token_usage("one") == sum(reply["agent_tokens"] for reply in replies)
    assert agent.prompt_token_usage("one") == sum(reply["prompt_tokens"] for reply in replies)
    assert agent.token_usage("one") > 0 and agent.prompt_token_usage("one") > 0
    if agent_type is AdvancedAgent:
        assert agent.compaction_count("one") > 1
        assert "Linh" in "\n".join(message_text(item) for item in model.seen[-1])
        assert len(model.seen[-1]) < 8
    else:
        assert agent.compaction_count("one") == 0
        assert len(model.seen[-1]) > 8
    agent.reply("bob", "two", "Mình tên gì?")
    assert "Linh" not in "\n".join(message_text(item) for item in model.seen[-1])


def test_live_profile_tools_are_scoped_to_current_user(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("langchain")
    from langchain_core.messages import AIMessage
    import agent_advanced

    responses = [
        AIMessage(content="", tool_calls=[{"name": "read_user_profile", "args": {}, "id": "read-1"}]),
        AIMessage(content="", tool_calls=[{"name": "edit_user_profile", "args": {"search_text": "Huế", "replacement": "Đà Nẵng"}, "id": "edit-1"}]),
        AIMessage(content="Đã sửa hồ sơ."),
    ]
    model = _fake_live_model(responses)
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: model)
    config = replace(make_config(tmp_path), offline=False, compact_threshold_tokens=10000)
    agent = AdvancedAgent(config)
    agent.profile_store.upsert_fact("alice", "location", "Huế")
    agent.profile_store.upsert_fact("bob", "location", "Cần Thơ")
    answer = agent.reply("alice", "one", "Hãy sửa hồ sơ nơi ở từ Huế sang Đà Nẵng.")
    assert answer["answer"] == "Đã sửa hồ sơ."
    assert agent.profile_store.facts("alice")["location"] == "Đà Nẵng"
    assert agent.profile_store.facts("bob")["location"] == "Cần Thơ"
    assert len(model.seen) == 3
    assert agent.token_usage("one") == answer["agent_tokens"]


def test_live_compact_keeps_tool_call_groups_valid(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("langchain")
    from langchain_core.messages import AIMessage, ToolMessage
    import agent_advanced

    responses = [AIMessage(content="", tool_calls=[{"name": "read_user_profile", "args": {}, "id": "read-1"}]),
                 AIMessage(content="Đã đọc hồ sơ.")]
    model = _fake_live_model(responses)
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: model)
    agent = AdvancedAgent(replace(make_config(tmp_path), offline=False))
    for index in range(4):
        agent.reply("alice", "one", "Mình tên là Linh. " + "Ngữ cảnh dài. " * 100)
    assert agent.compaction_count("one") > 0
    for messages in model.seen:
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage):
                preceding = [item for item in messages[:index] if isinstance(item, AIMessage)]
                assert preceding and any(call["id"] == message.tool_call_id for call in preceding[-1].tool_calls)


def test_offline_stress_style_has_three_bullets(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("alice", "one", "Mình tên là Linh. Mình muốn bạn trả lời ngắn gọn thành 3 bullet.")
    result = agent.reply("alice", "new", "Nhắc lại tên và style trả lời mình thích.")
    assert len(result["answer"].splitlines()) == 3
    assert all(line.startswith("- ") for line in result["answer"].splitlines())
    assert "Linh" in result["answer"] and "3 bullet" in result["answer"]


def test_personal_preference_statement_is_not_a_recall_question(tmp_path: Path) -> None:
    for agent_type in (BaselineAgent, AdvancedAgent):
        agent = agent_type(make_config(tmp_path), force_offline=True)
        result = agent.reply("alice", "one", "Mình thích Python và AI ứng dụng.")
        assert result["answer"] == "Mình đã ghi nhận thông tin bạn vừa chia sẻ."
        recall = agent.reply("alice", "one", "Nhắc lại mối quan tâm kỹ thuật của mình.")
        assert "Python" in recall["answer"] and "AI" in recall["answer"]


def test_topic_anchors_survive_compaction_but_stay_thread_local(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("alice", "one", "Dự án JAX-7 dùng Data Platform để triển khai. " + "Thông tin kỹ thuật dài. " * 80)
    agent.reply("alice", "one", "Tài liệu NIST nói về vận hành. " + "Báo cáo dài. " * 80)
    for _ in range(8):
        agent.reply("alice", "one", "Tiếp tục thảo luận. " + "Chi tiết vận hành. " * 80)
    answer = agent.reply("alice", "one", "Nhắc lại các chủ đề đã nói.")["answer"]
    assert all(topic in answer for topic in ("JAX-7", "Data Platform", "NIST"))
    assert "JAX-7" not in agent.profile_store.read_text("alice")
    fresh = agent.reply("alice", "new", "Nhắc lại các chủ đề đã nói.")["answer"]
    assert "JAX-7" not in fresh and "NIST" not in fresh
