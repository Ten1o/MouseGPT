"""Resolve model-service settings without network access or global mutations.

Stage integrations are intentionally separate: this module only loads and
validates configuration. It never logs credentials or sends API requests.
"""

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urlsplit


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / ".env.local"
ROLES = frozenset({"VISION", "EMBEDDING", "ANALYSIS", "AGENT", "PHENOTYPE"})
# These are OpenAI Chat Completions compatible base URLs, not a promise that
# every model on the service supports vision, embeddings, or tool calling.
PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
}
PROVIDERS = frozenset(PROVIDER_ENDPOINTS) | {"azure_openai", "openai_compatible"}
DEFAULT_MODELS = {
    "VISION": "gpt-4o-2024-05-13",
    "EMBEDDING": "text-embedding-3-large",
    "ANALYSIS": "gpt-4o",
    "AGENT": "gpt-4o",
    "PHENOTYPE": "gpt-4o",
}
LEGACY_AZURE_MODEL = {
    "VISION": "AZURE_OPENAI_DEPLOYMENT",
    "EMBEDDING": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "ANALYSIS": "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AGENT": "AZURE_OPENAI_CHAT_DEPLOYMENT",
}
LEGACY_AZURE_VERSION = {
    "EMBEDDING": "AZURE_OPENAI_EMBEDDING_API_VERSION",
    "ANALYSIS": "AZURE_OPENAI_CHAT_API_VERSION",
    "AGENT": "AZURE_OPENAI_CHAT_API_VERSION",
}
_KEY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OVERRIDE_FIELDS = frozenset({"provider", "endpoint", "api_key", "model", "api_version"})


@dataclass(frozen=True)
class ProviderConfig:
    role: str
    provider: str
    endpoint: str
    model: str
    api_version: str | None
    api_key: str = field(repr=False)

    def public_summary(self):
        """Return diagnostics safe to print; credentials are never included."""
        return {
            "role": self.role,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_version": self.api_version,
            "api_key_configured": bool(self.api_key),
        }


def http_request_config(config, operation):
    """Return an operation URL and auth headers for a resolved role."""
    if operation not in {"chat/completions", "embeddings"}:
        raise ValueError(f"Unsupported model operation: {operation}.")
    if config.provider == "azure_openai":
        deployment = quote(config.model, safe="")
        version = quote(config.api_version, safe="")
        url = (f"{config.endpoint}/openai/deployments/{deployment}/"
               f"{operation}?api-version={version}")
        auth = {"api-key": config.api_key}
    else:
        url = f"{config.endpoint}/{operation}"
        auth = {"Authorization": f"Bearer {config.api_key}"}
    return url, {"Content-Type": "application/json", **auth}


def sdk_client(config, *, max_retries=0, timeout=60):
    """Construct the SDK client only when an API-using workflow starts."""
    from openai import AzureOpenAI, OpenAI
    options = {"api_key": config.api_key, "max_retries": max_retries,
               "timeout": timeout}
    if config.provider == "azure_openai":
        return AzureOpenAI(azure_endpoint=config.endpoint,
                           api_version=config.api_version, **options)
    return OpenAI(base_url=config.endpoint, **options)


def read_env_file(path=CONFIG_FILE):
    """Read literal KEY=VALUE lines; no shell expansion or os.environ writes."""
    path = Path(path)
    if not path.is_file():
        return {}
    values = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _KEY_NAME.fullmatch(key):
            raise ValueError(f"Invalid configuration line {number} in {path.name}.")
        if key in values:
            raise ValueError(f"Duplicate configuration field {key} in {path.name}.")
        value = value.strip()
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed quoted value for {key} in {path.name}.")
            value = value[1:-1]
        values[key] = value
    return values


def _first(*values):
    return next((str(value).strip() for value in values
                 if value is not None and str(value).strip()), None)


def _field(role, name, overrides, environment, file_values, legacy=None):
    upper = name.upper()
    return _first(
        overrides.get(name),
        environment.get(f"MOUSEGPT_{role}_{upper}"),
        file_values.get(f"MOUSEGPT_{role}_{upper}"),
        environment.get(f"MOUSEGPT_{upper}"),
        file_values.get(f"MOUSEGPT_{upper}"),
        legacy,
    )


def _legacy_provider(environment):
    return "azure_openai" if environment.get("AZURE_OPENAI_ENDPOINT", "").strip() else "openai"


def _validate_endpoint(provider, endpoint):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("Endpoint must be an absolute HTTP(S) service base URL.")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote model-service endpoints must use HTTPS.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, a query, or a fragment.")
    path = parsed.path.rstrip("/")
    if provider in PROVIDER_ENDPOINTS and endpoint.rstrip("/") != PROVIDER_ENDPOINTS[provider]:
        raise ValueError(f"Provider '{provider}' requires {PROVIDER_ENDPOINTS[provider]}; "
                         "use 'openai_compatible' for a custom base URL.")
    if provider == "azure_openai" and path:
        raise ValueError("Azure endpoint must be the resource root, without /openai/deployments or operation paths.")
    if provider == "openai_compatible" and (path.endswith("/chat/completions") or path.endswith("/embeddings") or path.endswith("/models")):
        raise ValueError("Endpoint must be the service base URL, not a specific operation URL.")
    return endpoint.rstrip("/")


def load_config(role, *, overrides=None, env=None, env_file=CONFIG_FILE,
                require_key=True):
    """Resolve one role with CLI > role env > role file > shared env > shared file.

    Legacy environment variables are fallbacks when no new value exists.
    The function does not open a network connection or modify os.environ.
    """
    role = role.upper()
    if role not in ROLES:
        raise ValueError(f"Unknown model-service role: {role}.")
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides) - _OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"Unknown configuration override: {', '.join(sorted(unknown))}.")
    environment = os.environ if env is None else env
    file_values = read_env_file(env_file)
    provider = _field(role, "provider", overrides, environment, file_values,
                      _legacy_provider(environment))
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider for {role}: {provider}.")

    legacy_endpoint = (environment.get("AZURE_OPENAI_ENDPOINT") if provider == "azure_openai"
                       else PROVIDER_ENDPOINTS.get(provider))
    role_provider = _first(overrides.get("provider"), environment.get(f"MOUSEGPT_{role}_PROVIDER"),
                           file_values.get(f"MOUSEGPT_{role}_PROVIDER"))
    shared_provider = _first(environment.get("MOUSEGPT_PROVIDER"), file_values.get("MOUSEGPT_PROVIDER"),
                             _legacy_provider(environment))
    # A different role provider must not inherit another service's URL.
    if role_provider is not None and role_provider != shared_provider:
        endpoint = _first(overrides.get("endpoint"), environment.get(f"MOUSEGPT_{role}_ENDPOINT"),
                          file_values.get(f"MOUSEGPT_{role}_ENDPOINT"), legacy_endpoint)
    else:
        endpoint = _field(role, "endpoint", overrides, environment, file_values, legacy_endpoint)
    if not endpoint:
        raise ValueError(f"Set MOUSEGPT_ENDPOINT or MOUSEGPT_{role}_ENDPOINT.")
    endpoint = _validate_endpoint(provider, endpoint)

    role_endpoint = _first(overrides.get("endpoint"), environment.get(f"MOUSEGPT_{role}_ENDPOINT"),
                           file_values.get(f"MOUSEGPT_{role}_ENDPOINT"))
    shared_endpoint = _first(environment.get("MOUSEGPT_ENDPOINT"), file_values.get("MOUSEGPT_ENDPOINT"),
                             legacy_endpoint)
    changed_service = ((role_provider is not None and role_provider != shared_provider)
                       or (role_endpoint is not None and role_endpoint.rstrip("/") != (shared_endpoint or "").rstrip("/")))
    dedicated_key = _first(overrides.get("api_key"), environment.get(f"MOUSEGPT_{role}_API_KEY"),
                           file_values.get(f"MOUSEGPT_{role}_API_KEY"))
    if changed_service and require_key and not dedicated_key:
        raise ValueError(f"Set MOUSEGPT_{role}_API_KEY when overriding its provider or endpoint.")
    legacy_key = (environment.get("AZURE_OPENAI_API_KEY") if provider == "azure_openai" else
                  environment.get("OPENAI_API_KEY") if provider == "openai" else None)
    api_key = dedicated_key if changed_service else _field(
        role, "api_key", overrides, environment, file_values, legacy_key)
    if require_key and (not api_key or api_key.startswith("YOUR_")):
        raise ValueError(f"Set MOUSEGPT_{role}_API_KEY or MOUSEGPT_API_KEY for {role}.")

    legacy_model = (environment.get(LEGACY_AZURE_MODEL.get(role, ""))
                    if provider == "azure_openai" else
                    environment.get("OPENAI_CHAT_MODEL") if role in {"ANALYSIS", "AGENT"} else None)
    model = _field(role, "model", overrides, environment, file_values, legacy_model)
    if not model and provider == "openai":
        model = DEFAULT_MODELS[role]
    if not model:
        raise ValueError(f"Set MOUSEGPT_{role}_MODEL for {provider}.")

    legacy_version = None
    if provider == "azure_openai":
        legacy_version = _first(environment.get(LEGACY_AZURE_VERSION.get(role, "")),
                                environment.get("AZURE_OPENAI_API_VERSION"))
    api_version = _field(role, "api_version", overrides, environment, file_values, legacy_version)
    if provider == "azure_openai" and not api_version:
        raise ValueError(f"Set MOUSEGPT_{role}_API_VERSION for Azure OpenAI.")
    if provider != "azure_openai":
        api_version = None
    if role == "PHENOTYPE" and provider != "openai":
        raise ValueError("PHENOTYPE currently requires the official OpenAI Responses API with Code Interpreter.")
    if role == "EMBEDDING" and provider == "anthropic":
        raise ValueError("Anthropic does not provide an embedding endpoint; configure EMBEDDING with another provider.")
    return ProviderConfig(role, provider, endpoint, model, api_version, api_key or "")


def main():
    parser = argparse.ArgumentParser(description="Validate MouseGPT model-service settings without API calls.")
    parser.add_argument("--role", choices=sorted(ROLES) + ["ALL"], default="ALL",
                        type=str.upper)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE,
                        help="Configuration file (default: repository-root .env.local)")
    parser.add_argument("--allow-missing-key", action="store_true",
                        help="Inspect a template without a real API key")
    parser.add_argument("--ignore-environment", action="store_true",
                        help="Inspect file values without inherited environment variables")
    args = parser.parse_args()
    roles = sorted(ROLES - {"PHENOTYPE"}) if args.role == "ALL" else [args.role]
    try:
        summary = [load_config(role, env={} if args.ignore_environment else None,
                               env_file=args.config,
                               require_key=not args.allow_missing_key).public_summary()
                   for role in roles]
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
