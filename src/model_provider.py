from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


SUPPORTED_PROVIDERS = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {"anthorpic": "anthropic", "google": "gemini", "google-genai": "gemini",
               "openai-compatible": "custom", "open_router": "openrouter"}
    normalized = value.strip().lower()
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {value!r}. Choose from {sorted(SUPPORTED_PROVIDERS)}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Build only the selected provider; never make a request during construction."""
    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        raise ValueError("A model name is required for live mode")
    if provider == "custom" and not config.base_url:
        raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
    if provider not in {"ollama", "custom"} and not config.api_key:
        raise ValueError(f"An API key is required for live provider {provider!r}")

    classes = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "custom": ("langchain_openai", "ChatOpenAI"),
        "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    }
    module, class_name = classes[provider]
    try:
        model_class = getattr(import_module(module), class_name)
    except ImportError as exc:
        raise ImportError(f"Install {module.replace('_', '-')} to use {provider}") from exc
    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if provider in {"openai", "custom"}:
        kwargs["api_key"] = config.api_key or "not-needed"
        if config.base_url:
            kwargs["base_url"] = config.base_url
    elif provider == "gemini":
        kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
    elif provider == "anthropic":
        kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
    elif provider == "ollama":
        if config.base_url:
            kwargs["base_url"] = config.base_url
    else:
        kwargs["openrouter_api_key"] = config.api_key
        if config.base_url:
            kwargs["openrouter_api_base"] = config.base_url
    return model_class(**kwargs)
