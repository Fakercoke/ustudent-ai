"""Central env-var access. Read once, cached via lru_cache.

Provider-agnostic by design: we talk OpenAI-compatible APIs. Switching
between Groq / OpenAI / OpenRouter only changes .env, never the code.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider — defaults to Groq (free tier, 14,400 RPD).
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"

    # Cheap model for mechanical tasks (query rewriting). Empty = use llm_model.
    # On Groq there is no latency win (56ms vs 57ms median), so it stays off by
    # default; set it when moving to a per-token provider where 8B costs 5-10x
    # less than 70B.
    llm_small_model: str = ""

    # Service
    ai_service_port: int = 8000
    log_level: str = "INFO"

    # Lightweight operations dashboard.  The password is intentionally empty
    # by default: /ops stays disabled until a deployment supplies a secret.
    ops_db_path: str = "data/ops.sqlite3"
    ops_admin_username: str = "admin"
    ops_admin_password: str = ""
    ops_hash_salt: str = "ustudent-ai-local"
    ops_retention_days: int = 90
    ops_allow_insecure_http: bool = False
    ops_insecure_allowed_clients: str = "127.0.0.1,::1"
    ops_trusted_proxy_networks: str = ""
    ops_web_access_log_path: str = ""
    ops_web_log_max_bytes: int = 100_000_000
    ops_web_cache_seconds: int = 15
    ops_web_routes: str = "/,/login,/dashboard,/ai-chat"

    # Provider prices per one million tokens.  They are configuration rather
    # than hard-coded facts because model pricing changes independently of the
    # application.  A zero value means "show tokens, cost not configured".
    llm_cost_currency: str = "CNY"
    llm_cost_note: str = ""
    llm_input_cost_per_million: float = 0.0
    llm_cached_input_cost_per_million: float = 0.0
    llm_output_cost_per_million: float = 0.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
