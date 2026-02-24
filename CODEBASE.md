# Codebase Walkthrough

This document walks through the entire codebase so you can understand how the pieces fit together and start making changes confidently.

## How a Review Works End-to-End

The fastest way to understand the codebase is to follow a single `arch-review run` from CLI invocation to final output.

### 1. CLI parses arguments → `cli/run.py`

The Click command in `cli/run.py` validates input directories, loads the review profile (a YAML file), and reads shared infrastructure config from SSM Parameter Store via `load_shared_config()`. It then calls `_run_review()`, which creates a `ReviewOrchestrator` and calls `run_review()` on it.

### 2. Orchestrator builds agents → `review/orchestrator.py`

`ReviewOrchestrator.create()` is the factory that wires everything together:

- Creates two model variants via `create_model()`: a **standard model** (no reasoning) for simple tasks (requirements, questions, sparring, context condensation), and a **reasoning model** (with extended thinking) for architecture and review analysis. When `reasoning_level` is `"off"`, both are the same model instance.
- Creates all 5 agents, each with their own system prompt and tool set.
- Creates a `GuardrailsChecker` via `create_guardrails_checker()` if guardrail config is present in `SharedConfig`.

The constructor accepts pre-built agents via dependency injection, which is what tests use. The `create()` classmethod is the production path.

### 3. Orchestrator runs the 5-phase pipeline

`run_review()` executes the phases sequentially. Each phase follows the same pattern:

1. **Run the agent** — call the agent with a prompt, get raw output.
2. **Extract findings** — pass raw output through a context condenser (`review/context_condenser.py`) that uses a separate model call to produce structured bullet points. This prevents token overflow as context accumulates across phases.
3. **Check grounding** — optionally validate the condenser output against the raw output using Bedrock Guardrails (`review/grounding.py`), to catch hallucinated findings.
4. **Pass findings forward** — the extracted findings (not the raw output) become input to the next phase.

The phases in order:

| Phase | Agent | Input | Interactive? |
|-------|-------|-------|-------------|
| 1. Requirements | `RequirementsAnalyst` | Documents directory | No |
| 2. Architecture | `ArchitectureEvaluator` | Requirements findings + templates/diagrams/source | No |
| 2b. Defaults verification | Inline `DefaultsVerifier` agent | Architecture findings | No |
| 3. Questions | `QuestionAgent` | Architecture findings | Yes — `ask_user` tool |
| 4. Sparring | `SparringAgent` | Architecture + Q&A findings | Yes — `challenge_user` tool |
| 5. Review | `ReviewAgent` | All findings | No |

### 4. Results are saved → back in `cli/run.py`

`run_review()` returns a `ReviewResult` dataclass containing both the final review text and the full session transcript. The CLI writes these to `.arch-review/review.md` and `.arch-review/state.json`, extracts a verdict (PASS/FAIL/PASS WITH CONCERNS), and exits with the corresponding exit code.

## Package Layout

The package is organized by responsibility:

```
arch_sparring_agent/
├── agents/          # Agent factories and runners (one per review phase)
├── cli/             # Click commands and shared CLI helpers
├── config/          # Model registry and tuning constants
├── infra/           # AWS infrastructure lifecycle (Gateway, Policy Engine, etc.)
├── kb/              # Knowledge Base: scraper, S3 sync, infrastructure
├── review/          # Review pipeline: orchestrator, extraction, grounding
├── tools/           # File readers and analyzers given to agents as tools
├── profiles/        # Built-in YAML profile files
├── profiles.py      # Profile loading and resolution logic
├── state.py         # ReviewState dataclass for remediation persistence
└── exceptions.py    # Exception hierarchy
```

### Why some files are at the root

`profiles.py`, `state.py`, and `exceptions.py` live at the package root because they're cross-cutting — imported by multiple subpackages. Everything else is grouped by domain.

## Key Modules in Detail

### `agents/` — Agent Factories

Most files export two things:

- **`create_*_agent(...) -> Agent`** — Factory that builds an agent with its tools, system prompt, and model. Tools are defined as `@tool`-decorated closures inside the factory because they need access to the analyzer instances (e.g., `CloudFormationAnalyzer`) created there.
- **`run_*(agent, ...) -> str`** — Runner that invokes the agent with a specific prompt and returns the result as a string.

The exception is `requirements_agent.py`, which only exports `create_requirements_agent`. It has no `run_requirements` because the orchestrator calls the agent directly with a simple prompt and reads `str(result)` — no special prompt construction needed.

The agents don't know about each other. The orchestrator chains them by passing one agent's extracted findings as input to the next.

`kb_tool.py` is a shared factory that creates the `query_waf` tool used by both the architecture and review agents when a Knowledge Base is configured.

### `config/` — Model Registry and Tuning

Two files, re-exported through `config/__init__.py` so all imports use `from ..config import X`:

- **`models.py`** — The model registry (`SUPPORTED_MODELS`), `create_model()` factory, agent name constants (`AGENT_REQUIREMENTS`, etc.), reasoning configuration, and `REASONING_LEVELS` tuple. `create_model()` dispatches between Nova-style (`reasoningConfig`) and Claude-style (`adaptive` thinking) reasoning fields based on `reasoning_type` in the model config. Agent names live here because they're used by both agent factories and Cedar policy definitions — `config/` is the neutral location that avoids circular imports.
- **`tuning.py`** — Numeric constants (file size limits, chunk sizes, timeouts, thresholds) all overridable via environment variables using `_int_env()` / `_float_env()` helpers. Also defines the non-configurable `GUARDRAIL_NAME` constant.

### `review/` — The Review Pipeline

- **`orchestrator.py`** — `ReviewOrchestrator` coordinates the 5 phases, manages output capture for the session transcript, and runs the service-defaults verification step between Phase 2 and 3. All captured output passes through `_strip_redacted()` which removes `[REDACTED]` reasoning trace placeholders that Bedrock models emit when extended thinking is enabled.
- **`context_condenser.py`** — Structured extraction that converts verbose agent output into concise bullet points. Uses a model call (not regex) so it handles varied output formats. Falls back to chunked extraction for content that exceeds the context window.
- **`extraction.py`** — Markdown parsing for the final review output. Extracts gaps, risks, recommendations, and verdict from the review text to build a `ReviewState` for remediation mode. This is regex/string-based (not model-based) because the review output follows a known format.
- **`grounding.py`** — Post-hoc validation using Bedrock Guardrails' ApplyGuardrail API. `GuardrailsChecker` chunks content at `GROUNDING_CONTENT_CHUNK_SIZE` (API limit: 5K chars per content block), truncates sources at `GROUNDING_SOURCE_MAX_CHARS`, and reports the worst grounding score across all chunks. Gracefully degrades — guardrail unavailability never blocks a review. `create_guardrails_checker()` returns `None` when guardrail config is missing.

### `infra/` — AWS Infrastructure

All infrastructure is deploy-once, discover-via-SSM:

- **`shared_config.py`** — `SharedConfig` dataclass and SSM read/write. `arch-review deploy` writes resource IDs to `/arch-review/config` in SSM Parameter Store (as `SecureString`); `arch-review run` reads them. This is how the two commands communicate. The config includes `gateway_id`, `gateway_arn`, `policy_engine_id`, `region`, and optional `knowledge_base_id`, `kb_bucket_name`, `guardrail_id`, `guardrail_version`.
- **`gateway.py`** — Creates an AgentCore Gateway with Cognito OAuth authorization via the `bedrock-agentcore-starter-toolkit` SDK. Handles IAM propagation delays with exponential backoff polling.
- **`policy.py`** — Creates a Policy Engine and Cedar policies that restrict which tools each agent can use. The policy specs are defined in `_build_policy_specs()`. Policies are verified to reach `ACTIVE` status before proceeding. Existing policies are updated in place on conflict.
- **`guardrails.py`** — Creates a Bedrock Guardrail with a contextual grounding policy (configurable threshold via `GROUNDING_THRESHOLD`). The guardrail is used at review time by `review/grounding.py` to validate that condenser outputs are faithful to raw agent outputs. Idempotent — reuses an existing guardrail matching the name `ArchReviewGroundingGuardrail`.
- **`memory.py`** — Sets up AgentCore Memory for session persistence in remediation mode. Waits for memory to reach `ACTIVE` status before returning.
- **`polling.py`** — Generic `poll_until()` utility used across all infra modules for waiting on async AWS operations. Supports configurable interval, timeout, and exponential backoff.

### `tools/` — What Agents Can Do

Each tool class reads files from a user-provided directory and returns content as strings. Agents use these as `@tool`-decorated functions.

- **`cfn_analyzer.py`** — Lists and reads CloudFormation templates (YAML, JSON). Validates file size against `CFN_MAX_BYTES`.
- **`diagram_analyzer.py`** — Sends PNG/JPEG images to Bedrock's Converse API for multimodal analysis. Uses the model ID directly (not the Strands Agent) and Pillow for image validation. The model call is separate from the agent's model — it uses the raw `bedrock-runtime` client.
- **`document_parser.py`** — Reads markdown files with `python-frontmatter` support. Validates file size against `DOC_MAX_BYTES`.
- **`source_analyzer.py`** — Lists, reads, and searches Lambda/application source code. Caches the file listing. Validates file size against `SOURCE_MAX_BYTES`.
- **`kb_client.py`** — Wraps the Bedrock Agent Runtime Retrieve API for querying the WAF Knowledge Base with optional source filtering.
- **`common.py`** — Shared utilities: `validate_path()` (directory traversal protection), `validate_file_size()`, and `search_content()` (case-insensitive line search).

### `profiles.py` — Review Profiles

Profiles are YAML files with `name`, `description`, and `directives` (per-agent behavioral instructions appended to system prompts). Resolution order: project (`.arch-review/profiles/`) → user (`~/.config/arch-review/profiles/`) → built-in (packaged with the tool).

`load_profile()` returns a dict, and `get_directive(profile, agent_name)` extracts the relevant directive string. The profile dict flows opaquely through the orchestrator and agent factories — only `get_directive()` inspects its contents. Unknown keys in a profile generate a warning but don't cause errors.

Additional exported functions: `list_profiles()` (grouped by source: builtin/user/project), `get_profile_path()` (find a profile's file path), and `project_dir()` (CWD-relative project profiles directory).

### `cli/` — Commands

- **`__init__.py`** — Click group definition with explicit `add_command()` registration. Six commands: `run`, `deploy`, `destroy`, `remediate`, `profiles`, `kb`.
- **`common.py`** — Shared option decorators (`common_options` for `--region`/`--verbose`, `model_option` for `--model`), exit code constants (`EXIT_SUCCESS=0`, `EXIT_HIGH_RISK=1`, `EXIT_MEDIUM_RISK=2`, `EXIT_ERROR=3`), default file names, and helpers (`configure_logging`, `load_shared_config`, `get_verdict_and_exit_code`).
- **`run.py`** — The main review command. Archives previous outputs to `.arch-review/history/<date>/`, creates the orchestrator, runs the review, saves results and state. The `_archive_previous()` function handles history management with date-based directories.
- **`deploy.py`** — `deploy` and `destroy` commands for infrastructure lifecycle. Deploy creates Gateway, Policy Engine, Cedar policies, and Guardrail, then saves everything to SSM. Destroy reads config from SSM and tears down in reverse order (KB → Guardrail → Policy Engine → Gateway → SSM parameter).
- **`remediate.py`** — Loads `state.json` from a previous review and starts an interactive session with the `RemediationAgent`. Supports `--model`, `--no-output`, and `--output-dir` options.
- **`profiles.py`** — `list`, `show`, `create` subcommands for managing profiles.
- **`kb.py`** — `sync` subcommand that scrapes WAF docs, uploads to S3, and triggers KB ingestion.

### `exceptions.py` — Error Handling

```
ArchReviewError (base)
├── ConfigurationError    — AWS config, model access, credentials
├── PolicySetupError      — Policy engine, gateway, Cedar policies
├── ToolError             — File reading, diagram analysis, etc.
└── GuardrailSetupError   — Bedrock Guardrail lifecycle
```

Two exception tuples for catch groups:
- `MODEL_ERRORS` — `(ContextWindowOverflowException, MaxTokensReachedException, ClientError)` for model invocation failures.
- `AWS_ERRORS` — `(ClientError, BotoCoreError)` for general AWS service calls.

## Key Patterns

### Dependency Injection via Constructor + Factory Classmethod

`ReviewOrchestrator.__init__()` accepts pre-built agents. Tests inject mocks directly. Production code uses `ReviewOrchestrator.create()` which builds everything from directory paths and config.

### Idempotent Infrastructure

Every `setup_*` function handles "already exists" gracefully. The `ensure_resource()` helper in `polling.py` standardizes this: try to create, catch conflict, find existing. `arch-review deploy` is safe to run repeatedly.

### Region Flows Explicitly

Region is never mutated via `os.environ`. It's passed as a parameter from the CLI through to every AWS client creation. `DEFAULT_REGION` (from `AWS_REGION` env var, falling back to `eu-central-1`) is only used as a default value at function boundaries.

### Graceful Degradation

Optional components (guardrails, KB, session memory) log warnings and continue if unavailable. A review always completes even if supporting infrastructure is partially missing.

### Agent Names as Constants

Agent names in `config/models.py` (`AGENT_REQUIREMENTS`, etc.) are the single source of truth. They're used for Strands `Agent(name=...)` construction, Cedar policy definitions, and log messages.

## Common Tasks

### Adding a New Model

Add an entry to `SUPPORTED_MODELS` in `config/models.py` with the `model_id`, `description`, and `reasoning_type` (`"nova"` or `"adaptive"`). The CLI's `--model` choice list is generated from this dict automatically.

### Adding a New Agent Tool

Define a `@tool`-decorated function inside the relevant `create_*_agent()` factory in `agents/`, then add it to the `tools` list passed to `Agent()`. If the tool needs Cedar policy authorization, add it to the corresponding policy spec in `infra/policy.py` → `_build_policy_specs()`.

### Adding a New Review Profile

Create a YAML file in `arch_sparring_agent/profiles/` with `name`, `description`, and `directives` keys. The `directives` map agent names (`requirements`, `architecture`, `sparring`, `review`) to instruction strings that get appended to the agent's system prompt.

### Adding a New CLI Command

Create a new file in `cli/`, define a Click command, and register it with `cli.add_command()` in `cli/__init__.py`.

### Changing a Tuning Constant

All tuning constants in `config/tuning.py` are overridable via environment variables. To add a new one, use `_int_env("ARCH_REVIEW_YOUR_VAR", default_value)` and re-export it through `config/__init__.py`.

### Adjusting the Grounding Threshold

The contextual grounding check threshold is configurable via `ARCH_REVIEW_GROUNDING_THRESHOLD` (default: `0.7`). Lower values are more permissive; higher values require findings to be more closely grounded in the raw agent output.

## Testing

Tests are in `tests/` and use pytest. External dependencies (strands, boto3, bedrock_agentcore) are mocked at import time in `conftest.py` via `sys.modules` patching, so tests run without AWS credentials.

```bash
uv run pytest tests/ -v          # Run all tests
uv run pytest tests/test_tools.py # Run a specific test file
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run mypy arch_sparring_agent/  # Type check
```

The mock setup in `conftest.py` creates fake exception classes (`FakeClientError`, etc.) that support `isinstance` checks, so exception handling logic works correctly in tests.
