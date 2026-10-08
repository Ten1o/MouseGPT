"""Explicit, single-attempt OpenAI/Azure model execution (standard library only).

Importing this module, validating configuration and checking provider status do
not access the network. Callers must persist an attempt before execute_job and
must never silently retry uncertain outcomes. No model identifier is assumed.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


CONFIG_KEYS = {"provider", "model", "base_url", "api_key_env", "api_mode", "settings"}
SETTING_KEYS = {"max_output_tokens", "temperature", "reasoning_effort", "image_detail", "timeout", "stream"}
TASKS = {"inference", "ground_truth", "gt_check", "judge"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class ProviderExecutionError(RuntimeError):
    """A deliberately bounded failure safe to persist in run logs."""

    def __init__(self, code, message, *, retryable=False, uncertain=False, http_status=None):
        super().__init__(message)
        self.code = code
        self.retryable = bool(retryable)
        self.uncertain = bool(uncertain)
        self.http_status = http_status

    def as_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable,
                "uncertain": self.uncertain, "http_status": self.http_status,
                "automatic_retries": 0}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(label + " must be a lowercase SHA-256 digest")
    return value


def _endpoint(value, provider):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("base_url must be an HTTPS API base URL")
    try:
        parts = urlsplit(value)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError:
        raise ValueError("base_url must be an HTTPS API base URL") from None
    if (parts.scheme != "https" or not host or parts.username or parts.password
            or parts.query or parts.fragment or port not in (None, 443)
            or "\\" in value or any(ord(c) < 33 for c in value)):
        raise ValueError("base_url must be HTTPS on port 443 without credentials, query or fragment")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host) or ".." in host:
        raise ValueError("base_url needs a valid public DNS hostname")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("base_url needs a public DNS hostname, not an IP address")
    if "." not in host or host.endswith((".local", ".localhost", ".internal", ".invalid")):
        raise ValueError("base_url needs a public DNS hostname")
    path = parts.path.rstrip("/")
    if any(segment in (".", "..") for segment in path.split("/")) or "%" in path:
        raise ValueError("base_url path cannot contain traversal or encoded components")
    if provider == "azure":
        suffixes = (".openai.azure.com", ".services.ai.azure.com", ".openai.azure.us", ".openai.azure.cn")
        if not host.endswith(suffixes):
            raise ValueError("Azure needs an Azure OpenAI or Foundry resource hostname")
        if path in ("", "/"):
            path = "/openai/v1"
        if path != "/openai/v1":
            raise ValueError("Azure v1 base_url must end with /openai/v1")
    elif not path:
        path = "/v1"
    if path.endswith(("/responses", "/chat/completions")):
        raise ValueError("base_url must identify the API base, not a model operation")
    return urlunsplit(("https", host, path, "", ""))


def validate_config(config):
    """Return a normalized secret-free config, rejecting typo/credential fields."""
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError("Provider config accepts provider, model, base_url, api_key_env, api_mode and settings only")
    provider = config.get("provider")
    if provider not in ("openai", "azure"):
        raise ValueError("provider must be openai or azure")
    model = config.get("model")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", model):
        raise ValueError("model must be an explicit model or Azure deployment identifier")
    env = config.get("api_key_env", "OPENAI_API_KEY" if provider == "openai" else "AZURE_OPENAI_API_KEY")
    if not isinstance(env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", env):
        raise ValueError("api_key_env must name an environment variable; never enter a key here")
    mode = config.get("api_mode", "responses")
    if mode not in ("responses", "chat_completions"):
        raise ValueError("api_mode must be responses or chat_completions")
    base = config.get("base_url", "https://api.openai.com/v1" if provider == "openai" else None)
    settings = config.get("settings", {})
    if not isinstance(settings, dict) or set(settings) - SETTING_KEYS:
        raise ValueError("Unknown generation setting; only max_output_tokens, temperature, reasoning_effort, image_detail, timeout and stream are supported")
    settings = dict(settings)
    if "stream" in settings:
        if not isinstance(settings["stream"], bool):
            raise ValueError("stream must be a boolean")
        if mode != "responses":
            raise ValueError("stream is supported only for the Responses API")
    if "max_output_tokens" in settings:
        val = settings["max_output_tokens"]
        if isinstance(val, bool) or not isinstance(val, int) or not 1 <= val <= 1000000:
            raise ValueError("max_output_tokens must be an integer from 1 to 1000000")
    for key, low, high in (("temperature", 0, 2), ("timeout", 1, 3600)):
        if key in settings:
            val = settings[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not low <= val <= high:
                raise ValueError(key + " is outside its supported range")
    if settings.get("reasoning_effort", "medium") not in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
        raise ValueError("Unknown reasoning_effort")
    if settings.get("image_detail", "high") not in ("auto", "low", "high", "original"):
        raise ValueError("image_detail must be auto, low, high or original")
    settings.setdefault("timeout", 180)
    # Model-specific compatibility is checked by the provider. Never silently
    # remove/rewrite a requested parameter after a rejection.
    return {"provider": provider, "model": model, "base_url": _endpoint(base, provider),
            "api_key_env": env, "api_mode": mode, "settings": settings}


def provider_status(config):
    """No network call and no credential value in the returned status."""
    normalized = validate_config(config)
    return {"config": normalized, "credential_configured": bool(os.environ.get(normalized["api_key_env"], "").strip()),
            "execution_supported": True, "connection_verified": False, "network_requests": 0}


def _load_job(job, data_directory):
    if not isinstance(job, dict) or job.get("task") not in TASKS:
        raise ValueError("A supported versioned benchmark job is required")
    payload = job.get("payload_template")
    if not isinstance(payload, dict) or _sha(payload) != job.get("payload_sha256"):
        raise ValueError("Job payload checksum mismatch")
    _digest(job.get("input_contract_sha256"), "input_contract_sha256")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Job must include its frozen request messages")
    clean_messages, canonical_inputs = [], []
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user")
                or not isinstance(message.get("content"), str)):
            raise ValueError("Job template messages must be system/developer/user text")
        clean_messages.append({"role": message["role"], "content": message["content"]})
        if message["role"] == "user":
            try:
                obj = json.loads(message["content"])
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("canonical_input"), dict):
                canonical_inputs.append(obj["canonical_input"])
    if len(canonical_inputs) != 1:
        raise ValueError("Job must contain exactly one frozen canonical_input")
    canonical = canonical_inputs[0]
    if not isinstance(canonical.get("prompt"), str) or not canonical["prompt"].strip():
        raise ValueError("Frozen input must contain a nonempty prompt")
    from .recorded_inputs import STRUCTURED_PROTOCOLS
    logged = canonical.get("protocol_version") in STRUCTURED_PROTOCOLS
    detail = canonical.get("image_detail", "high")
    if detail not in ("auto", "low", "high", "original"):
        raise ValueError("Frozen image_detail is unsupported")
    if logged:
        from .providers import validate_logged_request
        from .recorded_inputs import input_semantics
        recorded_messages, bindings = validate_logged_request(canonical)
        if _sha(input_semantics(canonical)) != job["input_contract_sha256"]:
            raise ValueError("Logged input semantics do not match the frozen input hash")
        _digest(canonical.get("image_sha256"), "canonical review image sha256")
    else:
        if "input_images" in canonical or "request_messages" in canonical:
            raise ValueError("Recorded images and messages require the logged input protocol")
        bindings = [{"attachment_id": "canonical_image", "sha256": canonical.get("image_sha256"),
                     "image_detail": detail}]
    declared, manifests = payload.get("attachments"), job.get("attachment_manifest")
    if (not isinstance(declared, list) or not isinstance(manifests, list)
            or len(declared) != len(bindings) or len(manifests) != len(bindings)):
        raise ValueError("Each frozen image needs its ordered declaration and attachment manifest")
    root = Path(data_directory).resolve()
    images, image_meta = [], []
    for binding, spec, record in zip(bindings, declared, manifests):
        if not isinstance(spec, dict) or not isinstance(record, dict):
            raise ValueError("Invalid attachment manifest")
        identity, expected_hash = binding["attachment_id"], _digest(binding["sha256"], "image sha256")
        if any(item.get("attachment_id") != identity or item.get("sha256") != expected_hash for item in (spec, record)):
            raise ValueError("Image attachment identity, order or checksum mismatch")
        if logged and any(any(item.get(key) != value for key, value in binding.items()) for item in (spec, record)):
            raise ValueError("Logged image declaration or manifest differs from its frozen binding")
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("A local image path is required")
        path = (root / path_value).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            raise ValueError("Image must remain inside the benchmark data directory") from None
        if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Canonical image is missing or exceeds the image size limit")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Image attachment identity or checksum mismatch")
        if data.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        elif data.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            media_type = "image/webp"
        else:
            raise ValueError("Canonical input must be an encoded JPEG, PNG or WebP image")
        if logged and (binding["media_type"] != media_type or binding.get("size", len(data)) != len(data)):
            raise ValueError("Logged image media type or size does not match its original bytes")
        images.append(data)
        image_meta.append({"attachment_id": identity, "path": str(relative), "sha256": expected_hash,
                           "media_type": media_type, "size": len(data), "image_detail": binding["image_detail"]})
    if job["task"] == "inference":
        # Candidates receive the original request, not the GT/judge wrapper.
        velocity = canonical.get("velocity")
        if canonical.get("track") == "image_velocity":
            if not isinstance(velocity, dict) or "value" not in velocity or not velocity.get("units"):
                raise ValueError("image_velocity input requires a value and units")
        elif canonical.get("track") not in (None, "image_only") or velocity:
            raise ValueError("Velocity may only be sent on a frozen image_velocity track")
        if logged:
            if canonical.get("velocity_embedded_in_prompt") is not True:
                raise ValueError("Logged requests must explicitly preserve their embedded velocity text")
            clean_messages = recorded_messages
        else:
            system_prompt = canonical.get("system_prompt", "")
            if not isinstance(system_prompt, str):
                raise ValueError("Frozen system_prompt must be text")
            clean_messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            clean_messages.append({"role": "user", "content": canonical["prompt"]})
            if canonical.get("track") == "image_velocity":
                clean_messages.append({"role": "user", "content": "Velocity input:\n" + _json(
                    {k: velocity[k] for k in ("value", "units", "coordinate_system") if k in velocity})})
    elif not any("json" in message["content"].lower() for message in clean_messages):
        clean_messages.insert(0, {"role": "system", "content": "Return one JSON object conforming to the supplied output contract."})
    if not (logged and job["task"] == "inference"):
        users = [message for message in clean_messages if message["role"] == "user"]
        if not users:
            raise ValueError("Job must have at least one user message")
        users[0]["content"] = [{"type": "text", "text": users[0]["content"]}] + [
            {"type": "image_reference", "attachment_id": item["attachment_id"]} for item in image_meta]
    return clean_messages, images, image_meta, logged


def _payload(job, config, data_directory):
    messages, images, image_meta, logged = _load_job(job, data_directory)
    settings = config["settings"]
    if "image_detail" in settings and any(settings["image_detail"] != meta["image_detail"] for meta in image_meta):
        raise ValueError("Provider image_detail conflicts with the frozen input")
    image_lookup = {meta["attachment_id"]: (data, meta) for data, meta in zip(images, image_meta)}
    response_mode = config["api_mode"] == "responses"

    def convert_message(message, display=False):
        content = message["content"]
        # Chat completions can preserve the exact original string form. The
        # Responses adapter preserves roles and text but changes part types.
        if isinstance(content, str):
            if logged and job["task"] == "inference" and not response_mode:
                return {"role": message["role"], "content": content}
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "input_text" if response_mode else "text", "text": part["text"]})
            else:
                data, meta = image_lookup[part["attachment_id"]]
                detail = meta["image_detail"]
                if display:
                    value = {"image_reference": copy.deepcopy(meta), "detail": detail}
                    parts.append(dict(type="input_image", **value) if response_mode else {"type": "image_url", "image_url": value})
                else:
                    encoded = "data:" + meta["media_type"] + ";base64," + base64.b64encode(data).decode("ascii")
                    parts.append({"type": "input_image", "image_url": encoded, "detail": detail} if response_mode else
                                 {"type": "image_url", "image_url": {"url": encoded, "detail": detail}})
        return {"role": message["role"], "content": parts}

    request = {"model": config["model"], "store": False}
    message_key = "input" if response_mode else "messages"
    request[message_key] = [convert_message(message) for message in messages]
    if response_mode:
        if "stream" in settings:
            request["stream"] = settings["stream"]
        if job["task"] != "inference":
            request["text"] = {"format": {"type": "json_object"}}
        if "max_output_tokens" in settings:
            request["max_output_tokens"] = settings["max_output_tokens"]
        if "reasoning_effort" in settings:
            request["reasoning"] = {"effort": settings["reasoning_effort"]}
    else:
        if job["task"] != "inference":
            request["response_format"] = {"type": "json_object"}
        if "max_output_tokens" in settings:
            request["max_completion_tokens"] = settings["max_output_tokens"]
        if "reasoning_effort" in settings:
            request["reasoning_effort"] = settings["reasoning_effort"]
    if "temperature" in settings:
        request["temperature"] = settings["temperature"]
    display = copy.deepcopy(request)
    display[message_key] = [convert_message(message, display=True) for message in messages]
    # Existing single-image execution logs retain their historical object shape.
    return request, display, image_meta if logged else image_meta[0]


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_response_body(response, stream=False, on_event=None):
    """Read bounded JSON or Responses SSE without reconstructing model output.

    In streaming mode, ``on_event`` receives each decoded event object before
    terminal handling. Callers that persist events must redact credentials.
    Only a terminal event's complete response object is returned, including a
    terminal failed/incomplete response so callers can retain its raw receipt.
    A truncated stream, malformed event or standalone error is uncertain and is
    never converted into a successful partial answer.
    """
    if not stream:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ProviderExecutionError("response_too_large", "Provider response exceeded the local limit; completion may have occurred.", uncertain=True)
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ProviderExecutionError("invalid_response", "Provider returned an unreadable response; completion may have occurred.", uncertain=True) from None

    terminal_types = {"response.completed": "completed", "response.incomplete": "incomplete", "response.failed": "failed"}
    total_bytes = 0
    data_lines = []
    event_name = None
    while True:
        line = response.readline(MAX_RESPONSE_BYTES - total_bytes + 1)
        total_bytes += len(line)
        if total_bytes > MAX_RESPONSE_BYTES:
            raise ProviderExecutionError("response_too_large", "Provider response exceeded the local limit; completion may have occurred.", uncertain=True)
        if not line:
            raise ProviderExecutionError("incomplete_stream", "Provider stream ended without a terminal response; reconcile before explicitly retrying.", uncertain=True)
        try:
            line = line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            raise ProviderExecutionError("invalid_response", "Provider returned an unreadable stream event; completion may have occurred.", uncertain=True) from None
        if line:
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_name = value
            continue
        if not data_lines:
            event_name = None
            continue
        data = "\n".join(data_lines)
        data_lines = []
        if data == "[DONE]":
            raise ProviderExecutionError("incomplete_stream", "Provider stream ended without a terminal response; reconcile before explicitly retrying.", uncertain=True)
        try:
            event = json.loads(data)
            if not isinstance(event, dict):
                raise ValueError("Stream event must be an object")
            _json(event)
        except (TypeError, ValueError):
            raise ProviderExecutionError("invalid_response", "Provider returned a malformed stream event; completion may have occurred.", uncertain=True) from None
        if on_event is not None:
            on_event(copy.deepcopy(event))
        event_type = event.get("type", event_name)
        event_name = None
        if not isinstance(event_type, str):
            raise ProviderExecutionError("invalid_response", "Provider returned a malformed stream event type; completion may have occurred.", uncertain=True)
        if event_type in terminal_types:
            result = event.get("response")
            if not isinstance(result, dict) or result.get("status") != terminal_types[event_type]:
                raise ProviderExecutionError("invalid_response", "Provider terminal event did not contain its matching complete response object; completion may have occurred.", uncertain=True)
            return result
        if event_type == "error":
            raise ProviderExecutionError("provider_stream_error", "Provider emitted a stream error without a terminal response; reconcile before explicitly retrying.", uncertain=True)


def _http_transport(url, headers, payload, timeout):
    host = urlsplit(url).hostname
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise ProviderExecutionError("dns_failure", "Provider hostname could not be resolved; no request was sent.", retryable=True) from None
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ProviderExecutionError("unsafe_endpoint", "Provider endpoint resolved to a non-public address.")
    request = Request(url, data=_json(payload).encode("utf-8"), headers=headers, method="POST")
    opener = build_opener(_NoRedirects())
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as exc:
        # Provider error bodies can echo user payloads or credentials. Do not
        # expose/log them; HTTP status is sufficient for bounded classification.
        status = exc.code
        exc.close()
        return status, {}, {}
    with response:
        decoded = _read_response_body(response, stream=200 <= response.status < 300 and payload.get("stream") is True)
        return response.status, decoded, dict(response.headers)


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {str(key): ("[REDACTED]" if str(key).lower().replace("_", "-") in
                          ("api-key", "authorization", "access-token", "secret", "password") else _redact(item, secret))
                for key, item in value.items()}
    return value


def _extract_output(response, mode):
    if mode == "responses":
        if response.get("status") not in (None, "completed") or response.get("error"):
            raise ProviderExecutionError("incomplete_output", "Provider did not complete the response; inspect limits or refusal status before a new attempt.")
        parts = [part for item in response.get("output", []) if isinstance(item, dict) and item.get("type") == "message"
                 for part in item.get("content", []) if isinstance(part, dict)]
        if any(part.get("type") == "refusal" for part in parts):
            raise ProviderExecutionError("model_refusal", "Provider returned a refusal instead of a benchmark answer.")
        text = "\n".join(part["text"] for part in parts if part.get("type") == "output_text" and isinstance(part.get("text"), str))
        if not text and isinstance(response.get("output_text"), str):
            text = response["output_text"]
    else:
        choices = response.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderExecutionError("invalid_response", "Provider did not return exactly one candidate answer.", uncertain=True)
        choice = choices[0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise ProviderExecutionError("incomplete_output", "Provider stopped before a complete benchmark answer was available.")
        message = choice.get("message", {})
        if message.get("refusal"):
            raise ProviderExecutionError("model_refusal", "Provider returned a refusal instead of a benchmark answer.")
        text = message.get("content", "")
        if isinstance(text, list):
            text = "\n".join(item.get("text", "") for item in text if isinstance(item, dict) and item.get("type") == "text")
    if not isinstance(text, str) or not text.strip():
        raise ProviderExecutionError("empty_output", "Provider returned no usable answer.")
    return text


def execute_job(job, config, data_directory, transport=None):
    """Perform exactly one explicitly requested model call.

    The injectable transport receives (url, headers, payload, timeout) and returns
    (HTTP status, decoded JSON body, response headers), or a JSON body for tests.
    Inference preserves the frozen task prompt. The template's generic assistant
    wrapper is not sent to candidate models. Other tasks request JSON mode; final
    schema/GT/judgment validation remains the caller's responsibility.
    """
    config = validate_config(config)
    payload, display, image_meta = _payload(job, config, data_directory)
    key = os.environ.get(config["api_key_env"], "").strip()
    if not key:
        raise ProviderExecutionError("missing_credential", "The configured API key environment variable is empty; no request was sent.")
    if "\r" in key or "\n" in key:
        raise ProviderExecutionError("invalid_credential", "The configured API credential contains invalid header characters; no request was sent.")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if payload.get("stream") is True else "application/json", "User-Agent": "MouseBench/1"}
    headers["api-key" if config["provider"] == "azure" else "Authorization"] = key if config["provider"] == "azure" else "Bearer " + key
    endpoint = config["base_url"] + ("/responses" if config["api_mode"] == "responses" else "/chat/completions")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        result = (transport or _http_transport)(endpoint, headers, payload, config["settings"]["timeout"])
        if isinstance(result, dict):
            status, response, response_headers = 200, result, {}
        else:
            status, response, response_headers = result
    except ProviderExecutionError:
        raise
    except ssl.SSLCertVerificationError:
        raise ProviderExecutionError("tls_verification_failed", "TLS certificate verification failed; no trusted provider response was received.") from None
    except (TimeoutError, socket.timeout):
        raise ProviderExecutionError("transport_timeout", "Request timed out; completion may have occurred. Reconcile before explicitly retrying.", uncertain=True) from None
    except URLError as exc:
        if isinstance(exc.reason, socket.gaierror):
            raise ProviderExecutionError("dns_failure", "Provider hostname could not be resolved; no request was sent.", retryable=True) from None
        raise ProviderExecutionError("transport_failure", "The connection failed without a definitive response; reconcile before explicitly retrying.", uncertain=True) from None
    except Exception:
        raise ProviderExecutionError("transport_failure", "The execution transport failed without a definitive response; reconcile before explicitly retrying.", uncertain=True) from None
    if not isinstance(status, int) or not 200 <= status < 300:
        if status == 429:
            raise ProviderExecutionError("rate_limited", "Provider rejected the request due to a rate or quota limit. No automatic retry was made.", retryable=True, http_status=status)
        if isinstance(status, int) and (status >= 500 or status == 408):
            raise ProviderExecutionError("provider_failure", "Provider failure has an uncertain completion outcome; reconcile before explicitly retrying.", uncertain=True, http_status=status)
        if isinstance(status, int) and 300 <= status < 400:
            raise ProviderExecutionError("redirect_blocked", "Provider redirect was blocked; credentials were not forwarded.", http_status=status)
        raise ProviderExecutionError("request_rejected", "Provider rejected the request. Check model access, credentials and generation settings.", http_status=status)
    if not isinstance(response, dict):
        raise ProviderExecutionError("invalid_response", "Provider returned an unexpected response type; completion may have occurred.", uncertain=True)
    response = _redact(response, key)
    try:
        _json(response)
        text = _extract_output(response, config["api_mode"])
    except ProviderExecutionError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError):
        raise ProviderExecutionError("invalid_response", "Provider returned malformed response fields; completion may have occurred.", uncertain=True) from None
    try:
        parsed = json.loads(text)
        # Python accepts NaN by default; forbid non-finite values before import.
        _json(parsed)
    except (ValueError, TypeError):
        parsed = None
    log = {"kind": "native_execution", "log_kind": "native_execution", "input_sha256": job["input_contract_sha256"],
           "job_payload_sha256": job["payload_sha256"], "frame_id": job.get("frame_id"), "task": job["task"],
           "provider": config["provider"], "model": config["model"], "observed_model": response.get("model"),
           "api_mode": config["api_mode"], "endpoint": endpoint, "settings": config["settings"],
           "request_payload": display, "request_payload_sha256": _sha(payload), "image": image_meta,
           "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
           "elapsed_seconds": round(time.monotonic() - started, 6), "response_id": response.get("id"),
           "response_created_at": response.get("created_at", response.get("created")), "usage": response.get("usage"),
           "automatic_retries": 0}
    if isinstance(response_headers, dict):
        log["request_id"] = next((value for name, value in response_headers.items() if name.lower() in ("x-request-id", "apim-request-id")), None)
    return _redact({"output_text": text, "parsed_output": parsed, "raw_response": response, "request_log": log}, key)
