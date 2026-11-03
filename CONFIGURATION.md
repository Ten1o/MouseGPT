# Model providers and configuration

Use the [researcher quick start](README.md#quick-start-use-a-hosted-vlm) for a first run or the [agent guide](README_AGENTS.md) for automated execution. This reference defines the configuration for Stages 2–4. The shared loader is
implemented in `mousegpt_config.py`, and Stage 2/3/4 model requests now read
it. Copy `.env.example` to `.env.local` at the repository root. For the
default official OpenAI API, fill `MOUSEGPT_API_KEY`; the other fields already
have defaults. Existing `OPENAI_*` and `AZURE_OPENAI_*` environment variables
remain fallbacks when no new value takes precedence.

## Choose an entry point from the data you have

| Available input | Start here | Model-service role needed |
| --- | --- | --- |
| Synchronized videos, calibration, detector and pose weights | Stage 1, then prepare Stage 2 inputs | None for Stage 1 |
| JSONL prompts and images, including the small example | Stage 2 API client with `--backend api` | `VISION` |
| Paired camera images and matching velocity data | Stage 2 paired-camera description generation | `VISION` |
| Frame descriptions | Stage 3 embedding and HDF5 packaging | `EMBEDDING` |
| Compatible HDF5 session and matching image previews | Stage 4 interface | `AGENT`; `EMBEDDING` for search/classification; `ANALYSIS` for model-assisted clustering |

Prepared example sessions may start directly at Stage 4. A newly packaged session needs matching images to be shown there. Use `3_applications/create_h5.py --with-linkage` to include the hierarchy needed by clustering. This option computes average linkage with cosine distance and requires quadratic memory, so start with a small session. Intermediate data may skip earlier stages when they satisfy the selected stage's validation.

To inspect a session without processing it or making an API call, run
`python scripts/check_workflow.py --input /path/to/session`. This suggests a
starting stage from local filenames, checks whether that stage's model-service
role has a configured key, and points to its guide. It does not validate
checkpoint compatibility, full file contents, or provider access.

## File placement and secrets

The template lives at the repository root, alongside `1_keypoints/`, `2_VLM/`,
`3_applications/`, and `4_NLUI/`. Copy `.env.example` to `.env.local` there and
fill in the values for the roles you intend to use. `.env.local` is ignored by
Git. Leave real keys out of `.env.example`, command-line arguments, logs, and
screenshots. An environment variable can override a value in `.env.local`.

The shared loader finds the repository root from its own file location,
so commands can still be run from each stage's documented working directory.
It will read `.env.local` without changing `os.environ` globally.

The default configuration uses the official OpenAI API at
`https://api.openai.com/v1`. The template's model IDs match the current
defaults used by the relevant stage scripts. Users only need to provide a key
for the roles they run. Azure OpenAI and other compatible services are
explicit alternatives, not the default. Choosing Azure means setting the
endpoint with the Azure resource root, setting the deployment names, and
supplying the API versions required by those deployments.

## Common fields

| Field | Meaning |
| --- | --- |
| `MOUSEGPT_PROVIDER` | `openai`, `azure_openai`, `anthropic`, `gemini`, `groq`, `deepseek`, `mistral`, `openrouter`, `xai`, `siliconflow`, or `openai_compatible` |
| `MOUSEGPT_ENDPOINT` | Optional for built-in presets; required for Azure and custom compatible services. Must be a service base URL |
| `MOUSEGPT_API_KEY` | Shared key; never committed to the repository |
| `MOUSEGPT_<ROLE>_MODEL` | Model ID, or Azure deployment name |
| `MOUSEGPT_<ROLE>_API_VERSION` | Azure API version for that operation; unused for ordinary OpenAI-compatible endpoints |
| `MOUSEGPT_<ROLE>_PROVIDER`, `_ENDPOINT`, `_API_KEY` | Optional overrides for a role |

`ROLE` is `VISION`, `EMBEDDING`, `ANALYSIS`, or `AGENT`. `PHENOTYPE` is optional
and separate because that Stage 3 workflow uses Responses with Code
Interpreter; a chat-completions service alone does not satisfy it.

`VISION` generates descriptions from images in Stage 2. `EMBEDDING` generates
vectors in Stage 3 and query vectors in Stage 4. `ANALYSIS` covers the
model-assisted analysis/cluster calls. `AGENT` is the Stage 4 conversational
model and must support tool calling. These roles may use different services.
Stage 1 has no language-model role.

The optional local SWIFT inference server is separate from these external
model-service roles. If that server requires authentication, set
`MOUSEGPT_SERVING_API_KEY` in the same `.env.local`; its clients do not use or
forward `MOUSEGPT_API_KEY` automatically.

For each role, value priority is: an explicit supported CLI override,
then that role's environment variable, then that role's `.env.local` value,
then the shared environment variable, then the shared `.env.local` value.
Required values that remain empty will cause a targeted preflight error before
billable requests. Existing legacy environment variables will be accepted as
fallbacks during migration, without silently overriding explicit new fields.
When a role selects a different provider or endpoint, it must also set its
own `MOUSEGPT_<ROLE>_API_KEY`; the shared key is never forwarded to that other
service implicitly. A built-in provider selects its documented base URL and
rejects a stale endpoint from another provider. For custom gateways, use
`openai_compatible` and supply its URL. Legacy `OPENAI_API_KEY` is only a
fallback for `openai`; it is not sent to another provider.

The loader can be inspected offline from the repository root:

```bash
python -m mousegpt_config --config .env.example --allow-missing-key --ignore-environment
```

This prints a redacted summary for the four regular roles using only template
values, so an existing shell key cannot influence this particular check. To validate a
filled `.env.local`, run `python -m mousegpt_config`; it checks presence and
syntax but does not contact the provider or verify that a key has access.
Use `--role VISION`, `EMBEDDING`, `ANALYSIS`, `AGENT`, or `PHENOTYPE` to inspect
one role. A missing key is permitted only with `--allow-missing-key`.

For an explicit, single-request API smoke test, run the following from the
repository root after checking that the selected key is safe to use:

```bash
python scripts/download_assets.py vlm-demo
python scripts/download_assets.py vlm-demo --verify
python scripts/smoke_model_api.py --role AGENT
python scripts/smoke_model_api.py --role EMBEDDING
python scripts/smoke_model_api.py --role VISION
```

These commands are dry runs. Add `--live` to one command to send exactly one
potentially billable request for that role. The vision check uses one small
JPEG from the downloaded `vlm-demo` bundle by default; `--image PATH` selects another JPEG/PNG no larger than
1 MB. This checks basic chat, vector length, or image acceptance, not the full
Stage 2/3/4 pipeline or Agent tool calling. It never prints the configured key.

## Endpoint examples

When using a configuration snippet below, edit existing entries and add only missing names in `.env.local`. Each setting must appear once; do not append a second model or API-key entry already present in the template.

| Provider | `MOUSEGPT_ENDPOINT` example | Model field means |
| --- | --- | --- |
| `openai` | `https://api.openai.com/v1` | OpenAI model ID |
| `azure_openai` | `https://RESOURCE.openai.azure.com/` | Azure deployment name |
| `anthropic` (Claude) | `https://api.anthropic.com/v1` | Claude model ID; no Anthropic embeddings |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` | Gemini model ID |
| `groq` | `https://api.groq.com/openai/v1` | Groq-served model ID |
| `deepseek` | `https://api.deepseek.com` | DeepSeek model ID |
| `mistral` | `https://api.mistral.ai/v1` | Mistral model ID |
| `openrouter` | `https://openrouter.ai/api/v1` | Router model ID, including provider prefix |
| `xai` | `https://api.x.ai/v1` | xAI model ID |
| `siliconflow` | `https://api.siliconflow.cn/v1` | SiliconFlow model ID |
| `openai_compatible` | `https://provider.example/v1` | Provider model ID |

The service base URL must not include `/models`, `/embeddings`,
`/chat/completions`, `/openai/deployments/...`, or an Azure `api-version` query.
An OpenAI-compatible provider is not guaranteed to offer vision, embeddings,
tool calling, or Responses. Each selected workflow must check the capabilities
it actually uses. A connection test may incur a small provider charge.

For a built-in provider, change `MOUSEGPT_PROVIDER`, `MOUSEGPT_API_KEY`, and
the model fields for the roles you use; leave `MOUSEGPT_ENDPOINT` unset.
Alternatively, override only the role you want to replace. For example, a
third-party VLM with the default OpenAI embedding service can use:

```dotenv
MOUSEGPT_PROVIDER=openai
MOUSEGPT_API_KEY=YOUR_OPENAI_KEY
MOUSEGPT_EMBEDDING_MODEL=text-embedding-3-large
MOUSEGPT_VISION_PROVIDER=openai_compatible
MOUSEGPT_VISION_ENDPOINT=https://YOUR_PROVIDER_BASE_URL/v1
MOUSEGPT_VISION_API_KEY=YOUR_VISION_PROVIDER_KEY
MOUSEGPT_VISION_MODEL=YOUR_IMAGE_CAPABLE_MODEL_ID
```

Run this configuration with `client_test.py --backend api` as shown in the
[quick start](README.md#quick-start-use-a-hosted-vlm). Do not use the SWIFT
backend for a generic hosted endpoint. For a Claude-powered conversational
agent and OpenAI embeddings, the role overrides are:

```dotenv
MOUSEGPT_PROVIDER=openai
MOUSEGPT_API_KEY=YOUR_OPENAI_KEY
MOUSEGPT_AGENT_PROVIDER=anthropic
MOUSEGPT_AGENT_API_KEY=YOUR_ANTHROPIC_KEY
MOUSEGPT_AGENT_MODEL=YOUR_CLAUDE_MODEL_ID
```

The role's provider preset supplies its URL without inheriting the shared
OpenAI URL. Model IDs are intentionally not guessed for non-OpenAI providers:
choose an accessible model that supports the required capability. Anthropic
does not provide its own embedding endpoint, so `EMBEDDING=anthropic` is
rejected. `PHENOTYPE` remains official-OpenAI-only because that workflow uses
Responses with Code Interpreter, beyond ordinary chat compatibility. The
Stage 4 Agent also requires tool calling; a basic chat smoke test does not
prove tool-call support. Non-OpenAI providers may need provider-specific
adjustments to prompts or response formats after a live end-to-end test.
Qwen/DashScope also has an OpenAI-compatible mode, but its recommended URL
depends on region and workspace; select `openai_compatible` and enter the
base URL for your own account. The same custom option covers other gateways.

## Vector compatibility and model changes

Stage 3 accepts embeddings of any consistent positive dimension. Each vector must be finite and nonzero. Its cache sidecar records the service, endpoint, model/deployment, API version, settings, dimension, description request hash, and vector hash. Session packaging checks those records and preserves the identity in the HDF5 `features_des` metadata.

Stage 4 search and classification validate this identity against the current `EMBEDDING` role before making a query request. A different provider, endpoint, model, API version, or relevant setting requires regenerating embeddings and packaging the session again. Matching vector lengths do not make two embedding spaces compatible. Use stable model IDs and rebuild vectors if a provider changes the model behind a deployment alias; unchanged configuration text cannot detect that server-side change.

Old vectors without verified metadata are not silently assigned the current model. The exact published `nlui-demo` HDF5 is recognized by its verified file hash and original official OpenAI `text-embedding-3-large` identity. Other legacy sessions must be regenerated from their descriptions. Copying a model name into metadata does not establish provenance.

Changing the `VISION` model requires a new API-client output directory; unchanged completed records can resume. Paired-camera annotation caches and embedding caches validate their recorded inputs before reuse. Clustering caches include the selected analysis service and model, so changing `ANALYSIS` does not reuse another model's labels.

## Request compatibility

All provider presets use an OpenAI-compatible API surface; a preset is not a guarantee that every model offered by that service supports the selected role. Choose an image-capable model for VISION, an embedding model for EMBEDDING, and a tool-capable chat model for AGENT. Optional PHENOTYPE remains a separate OpenAI Responses workflow.

Official OpenAI and Azure chat requests use `max_completion_tokens`; other compatible services use `max_tokens`. Sampling options are omitted where the selected request path requires that. Providers can expose additional model-specific constraints, and completion budgets may include reasoning tokens. Test the actual stage with a small input before scaling up. If a model rejects an option or cannot return the expected schema, inspect the failure and select a compatible configuration; do not repeatedly retry an unchanged rejected request.

## What the checks establish

`python -m mousegpt_config` checks local configuration, not credentials or
provider capabilities. `scripts/smoke_model_api.py --live` makes one request
for the selected role, but does not exercise a complete stage. To validate a
workflow, run the relevant stage commands with a small representative input
and inspect their outputs. A successful run with one provider and model does
not establish that every listed provider supports vision, embeddings, tool
calling, or the optional Responses/Code Interpreter workflow.
