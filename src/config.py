from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    offline: bool = True

    def __post_init__(self) -> None:
        if self.compact_threshold_tokens <= 0:
            raise ValueError("COMPACT_THRESHOLD_TOKENS must be positive")
        if self.compact_keep_messages < 1:
            raise ValueError("COMPACT_KEEP_MESSAGES must be at least 1")


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Environment overrides .env. Offline is the explicit default."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    env_file = root / ".env"
    values: dict[str, str | None] = {}
    if env_file.exists():
        from dotenv import dotenv_values
        values.update(dotenv_values(env_file))
    values.update(os.environ)

    def setting(key: str, default: str = "") -> str:
        return values.get(key) or default

    defaults = {"openai": "gpt-4o-mini", "custom": "local-model", "gemini": "gemini-2.5-flash",
                "anthropic": "claude-sonnet-4-5", "ollama": "llama3.2", "openrouter": "openai/gpt-4o-mini"}
    keys = {"openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY", "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "ollama": "OLLAMA_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    urls = {provider: provider.upper() + "_BASE_URL" for provider in defaults}

    def provider_config(prefix: str = "", fallback: ProviderConfig | None = None) -> ProviderConfig:
        provider = normalize_provider(setting(prefix + "LLM_PROVIDER", fallback.provider if fallback else "openai"))
        same_provider = fallback is not None and provider == fallback.provider
        return ProviderConfig(
            provider=provider,
            model_name=setting(prefix + "LLM_MODEL", fallback.model_name if same_provider else defaults[provider]),
            temperature=float(setting(prefix + "LLM_TEMPERATURE", str(fallback.temperature) if same_provider else "0")),
            api_key=setting(prefix + keys[provider], setting(keys[provider])) or None,
            base_url=setting(prefix + urls[provider], setting(urls[provider])) or None,
        )

    offline_text = setting("LAB_OFFLINE", "true").lower()
    if offline_text not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("LAB_OFFLINE must be true or false")
    state = Path(setting("STATE_DIR", "state")).expanduser()
    state = state if state.is_absolute() else root / state
    model = provider_config()
    config = LabConfig(root, root / "data", state, int(setting("COMPACT_THRESHOLD_TOKENS", "1000")),
                       int(setting("COMPACT_KEEP_MESSAGES", "4")), model, provider_config("JUDGE_", model),
                       offline_text in {"true", "1", "yes"})
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
