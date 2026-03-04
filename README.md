# Architecture Review Sparring Partner

Multi-agent system for architecture reviews. Analyzes requirements documents, CloudFormation templates, architecture diagrams, and source code, then challenges architectural decisions through interactive sparring.

## Features

- **5-phase review process**: Requirements → Architecture → Questions → Sparring → Final Review
- **Interactive sparring**: Challenges each gap individually with follow-up rounds, pushing back on weak justifications before classifying outcomes
- **Remediation mode**: Discuss and resolve findings from previous reviews with session memory
- **CDK support**: Works with CloudFormation templates and CDK synthesized output (`cdk.out/`)
- **Multimodal analysis**: Analyzes architecture diagrams (PNG, JPEG) via Bedrock Converse API
- **Contextual grounding**: Post-hoc validation of extracted findings via Bedrock Guardrails to catch hallucinations
- **Service defaults verification**: Filters false positives by checking flagged gaps against AWS service defaults
- **Full session export**: Saves complete review session to markdown
- **Review profiles**: Customizable behavioral profiles (strict, lightweight, or your own)
- **WAF Knowledge Base**: Optional RAG-powered retrieval of AWS Well-Architected Framework best practices
- **Shared infrastructure**: Deploy once per AWS account, shared across team members
- **OpenTelemetry tracing**: Optional `otel` extra for Strands SDK's built-in agent/tool tracing

## Prerequisites

- Python 3.11+
- AWS credentials configured
- Nova 2 Lite model access in Bedrock console

## Installation

```bash
pip install arch-sparring-agent

# With Strands SDK OpenTelemetry tracing (optional)
pip install arch-sparring-agent[otel]
```

The package also supports `python -m arch_sparring_agent`.

## Quick Start

```bash
# 1. Deploy shared infrastructure (once per account)
arch-review deploy

# 2. Run an architecture review
arch-review run \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams

# 3. Discuss and resolve findings
arch-review remediate
```

## Commands

### `arch-review deploy`

Deploy shared infrastructure to an AWS account. Creates the Gateway, Policy Engine, Cedar policies, and a contextual grounding Guardrail. Stores resource IDs in SSM Parameter Store so `arch-review run` discovers them automatically.

```bash
arch-review deploy
arch-review deploy --with-kb        # Also provision a WAF Knowledge Base
arch-review deploy --region us-east-1
```

Idempotent — safe to run repeatedly.

### `arch-review destroy`

Tear down all shared infrastructure including Gateway, Policy Engine, Guardrail, Knowledge Base (if present), and SSM parameter.

```bash
arch-review destroy --confirm
```

### `arch-review run`

Run an interactive architecture review.

```bash
# Basic usage
arch-review run \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams

# With source code analysis
arch-review run \
    --documents-dir ./docs \
    --templates-dir ./cdk.out \
    --diagrams-dir ./diagrams \
    --source-dir ./src/lambdas

# Use a different profile
arch-review run --profile strict \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams
```

### `arch-review remediate`

Discuss and resolve findings from a previous review:

```bash
arch-review remediate
arch-review remediate --model opus-4.6
arch-review remediate --no-output       # Don't save notes to file
```

- Loads gaps/risks from `.arch-review/state.json`
- Continues conversations across sessions via AgentCore Memory
- Saves notes to `.arch-review/remediation-notes.md`

### `arch-review profiles`

Manage behavioral profiles that control how agents conduct reviews.

```bash
arch-review profiles list              # List all available profiles
arch-review profiles show strict       # Display a profile's contents
arch-review profiles create myprofile  # Create a new profile from the default
```

### `arch-review kb`

Manage the WAF Knowledge Base (requires `deploy --with-kb` first).

```bash
arch-review kb sync                    # Scrape WAF docs, upload to S3, trigger ingestion
arch-review kb sync --content-dir ./my-waf-content
```

## Review Profiles

Profiles control agent behavior — how strict the review is, what justifications are accepted, and how findings are reported. Three built-in profiles are included:

| Profile       | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| `default`     | Balanced review — thorough but pragmatic (2 rounds for high-severity gaps) |
| `strict`      | Low tolerance for gaps, demands evidence (up to 3 rounds per gap) |
| `lightweight` | Pragmatic for prototypes and demos (1 round per gap)              |

```bash
arch-review run --profile strict --documents-dir ./docs --templates-dir ./templates --diagrams-dir ./diagrams
```

### Custom Profiles

Profiles are YAML files searched in order:

1. **Project-level**: `.arch-review/profiles/` (checked first)
2. **User-level**: `~/.config/arch-review/profiles/`
3. **Built-in**: Packaged with the tool

Create a custom profile:

```bash
arch-review profiles create myprofile           # Copies from default
arch-review profiles create myprofile --from strict  # Copies from strict
```

Each profile is a complete, standalone specification — no layering or overrides. Edit the generated YAML to adjust behavioral directives for each agent. Profiles also control sparring depth via `settings.sparring.max_rounds` (keyed by severity: `high`, `medium`, `low`).

## WAF Knowledge Base

The optional Knowledge Base provides agents with AWS Well-Architected Framework best practices via RAG, improving the quality and accuracy of architecture reviews.

```bash
# Deploy with KB
arch-review deploy --with-kb

# Scrape all 6 WAF pillars + official lenses, upload to S3, and trigger ingestion
arch-review kb sync
```

Once synced, the architecture and review agents automatically query the KB for relevant best practices during analysis. Re-run `kb sync` periodically to pick up AWS documentation updates.

The KB uses **S3 Vectors** as the vector store (cost-effective, no OpenSearch Serverless overhead) and **Amazon Titan Embed Text v2** for embeddings.

## Options

### `run` Options

| Option                    | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| `--documents-dir`         | Directory with markdown requirements/constraints          |
| `--templates-dir`         | CloudFormation templates or `cdk.out/` directory          |
| `--diagrams-dir`          | Architecture diagrams (PNG, JPEG)                         |
| `--source-dir`            | Lambda/application source code (optional)                 |
| `--output-dir`            | Output directory (default: `.arch-review`)                |
| `--profile`               | Review profile: `default`, `strict`, `lightweight`, or custom |
| `--no-history`            | Don't archive previous reviews                            |
| `--no-state`              | Don't save state file after review                        |
| `--reasoning-level`       | Reasoning effort: off, low, medium, high (default: low)   |
| `-v`, `--verbose`         | Show detailed output (policy setup, debug info)           |
| `--model`                 | Model: `nova-2-lite` or `opus-4.6` (default: nova-2-lite) |
| `--region`                | AWS region (default: eu-central-1)                        |

### `deploy` Options

| Option                | Description                                        |
| --------------------- | -------------------------------------------------- |
| `--region`            | AWS region (default: eu-central-1)                 |
| `--gateway-name`      | Name for the Gateway resource                      |
| `--policy-engine-name`| Name for the Policy Engine resource                |
| `--with-kb`           | Also provision a WAF Knowledge Base                |
| `-v`, `--verbose`     | Verbose output                                     |

### `destroy` Options

| Option                | Description                                        |
| --------------------- | -------------------------------------------------- |
| `--region`            | AWS region (default: eu-central-1)                 |
| `--confirm`           | Required to actually destroy resources             |
| `-v`, `--verbose`     | Verbose output                                     |

### `remediate` Options

| Option                | Description                                        |
| --------------------- | -------------------------------------------------- |
| `--output-dir`        | Output directory (default: `.arch-review`)         |
| `--no-output`         | Don't save remediation notes to file               |
| `--model`             | Model to use (default: nova-2-lite)                |
| `--region`            | AWS region (default: eu-central-1)                 |
| `-v`, `--verbose`     | Verbose output                                     |

### `profiles` Subcommands

| Subcommand            | Description                                        |
| --------------------- | -------------------------------------------------- |
| `list`                | List all available profiles (built-in, user, project) |
| `show <name>`         | Display a profile's YAML contents                  |
| `create <name>`       | Create a new profile (copies from `--from`, default: `default`) |

## Supported Models

The `--model` flag accepts a short name from the curated model registry.
Only models with 1M context windows are supported to ensure reliable full-project reviews.

| Short Name      | Model                     | Context | `--model` value |
| --------------- | ------------------------- | ------- | --------------- |
| Nova 2 Lite     | Amazon Nova 2 Lite        | 1M      | `nova-2-lite`   |
| Claude Opus 4.6 | Anthropic Claude Opus 4.6 | 1M      | `opus-4.6`      |

**Examples:**

```bash
# Default (Nova 2 Lite with low reasoning)
arch-review run --documents-dir ./docs --templates-dir ./cdk.out --diagrams-dir ./diagrams

# Claude Opus 4.6 with medium reasoning
arch-review run --model opus-4.6 --reasoning-level medium \
    --documents-dir ./docs --templates-dir ./cdk.out --diagrams-dir ./diagrams
```

**Reasoning levels:**

- `off` -- disable extended thinking entirely
- `low` -- minimal reasoning (default, fastest)
- `medium` -- balanced reasoning
- `high` -- maximum reasoning (slowest, best quality)

### Model Quotas & Cost

AWS Bedrock enforces **daily token quotas** per model at the account level. These quotas are shared across all users and workloads on the same AWS account.

| Model         | Cross-Region Daily Quota | Relative Cost |
| ------------- | ------------------------ | ------------- |
| `nova-2-lite` | ~432M tokens             | Low           |
| `opus-4.6`    | ~2.6M tokens             | High          |

> **Warning:** Opus 4.6 has a very low default daily token quota (~2.6M tokens for cross-region inference). A single architecture review involves multiple agent calls (requirements, architecture, questions, sparring, final review), each consuming tokens. You may only get **1–2 reviews per day** before hitting the limit.
>
> Additionally, Opus 4.6 uses [adaptive thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html) with automatic interleaved thinking between tool calls. Thinking tokens are billed as output tokens ([docs](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html#claude-messages-extended-thinking-cost)) and most Claude models apply a [5x burndown rate](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html) on output tokens (1 output token = 5 tokens from your quota). This significantly amplifies quota consumption.
>
> For Nova 2 Lite, reasoning tokens are also charged even though reasoning content is redacted ([docs](https://docs.aws.amazon.com/nova/latest/nova2-userguide/reasoning-capabilities.html)).
>
> For iterative development and frequent reviews, **`nova-2-lite` (the default) is strongly recommended**. Reserve `opus-4.6` for cases where higher reasoning quality is critical.
>
> These quotas are marked as non-adjustable in AWS Service Quotas. Contact AWS Support to request an increase.

## Environment Variables

All options can be set via environment variables:

| Variable                      | Description                              |
| ----------------------------- | ---------------------------------------- |
| `ARCH_REVIEW_DOCUMENTS_DIR`   | Documents directory                      |
| `ARCH_REVIEW_TEMPLATES_DIR`   | Templates directory                      |
| `ARCH_REVIEW_DIAGRAMS_DIR`    | Diagrams directory                       |
| `ARCH_REVIEW_SOURCE_DIR`      | Source code directory                    |
| `ARCH_REVIEW_OUTPUT_DIR`      | Output directory                         |
| `ARCH_REVIEW_MODEL`           | Model short name (nova-2-lite, opus-4.6) |
| `ARCH_REVIEW_REASONING_LEVEL` | Reasoning effort: off, low, medium, high |
| `AWS_REGION`                  | AWS region                               |

## Exit Codes

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| 0    | PASS - no significant issues found               |
| 1    | FAIL - critical issues found                     |
| 2    | PASS WITH CONCERNS - gaps found but non-critical |
| 3    | Error during execution                           |

## AWS Credentials

The tool uses the standard AWS credential chain.

### Local Development

Configure credentials using any standard method:

```bash
# Option 1: AWS CLI profile
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# Option 3: AWS SSO
aws sso login --profile my-profile
export AWS_PROFILE=my-profile
```

### Required IAM Permissions

> **Note**: For production use, scope `Resource` to your specific account/region ARNs.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockModelAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:Converse",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockGuardrails",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateGuardrail",
        "bedrock:ListGuardrails",
        "bedrock:GetGuardrail",
        "bedrock:DeleteGuardrail",
        "bedrock-runtime:ApplyGuardrail"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AgentCorePolicyAndGateway",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreatePolicyEngine",
        "bedrock-agentcore:DeletePolicyEngine",
        "bedrock-agentcore:ListPolicyEngines",
        "bedrock-agentcore:CreatePolicy",
        "bedrock-agentcore:DeletePolicy",
        "bedrock-agentcore:UpdatePolicy",
        "bedrock-agentcore:GetPolicy",
        "bedrock-agentcore:ListPolicies",
        "bedrock-agentcore:CreateGateway",
        "bedrock-agentcore:DeleteGateway",
        "bedrock-agentcore:GetGateway",
        "bedrock-agentcore:UpdateGateway",
        "bedrock-agentcore:ListGateways"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AgentCoreMemory",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateMemory",
        "bedrock-agentcore:ListMemories"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SSMConfig",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:PutParameter",
        "ssm:DeleteParameter"
      ],
      "Resource": "arn:aws:ssm:*:*:parameter/arch-review/*"
    },
    {
      "Sid": "CallerIdentity",
      "Effect": "Allow",
      "Action": "sts:GetCallerIdentity",
      "Resource": "*"
    },
    {
      "Sid": "KnowledgeBaseOptional",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateKnowledgeBase",
        "bedrock:DeleteKnowledgeBase",
        "bedrock:ListKnowledgeBases",
        "bedrock:CreateDataSource",
        "bedrock:DeleteDataSource",
        "bedrock:ListDataSources",
        "bedrock:StartIngestionJob",
        "bedrock:GetIngestionJob",
        "bedrock-agent-runtime:Retrieve",
        "s3:CreateBucket",
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject",
        "s3:DeleteBucket",
        "s3vectors:CreateVectorBucket",
        "s3vectors:DeleteVectorBucket",
        "s3vectors:ListVectorBuckets",
        "s3vectors:CreateIndex",
        "s3vectors:DeleteIndex",
        "s3vectors:ListIndexes",
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy"
      ],
      "Resource": "*"
    }
  ]
}
```

The `KnowledgeBaseOptional` statement is only needed if you use `deploy --with-kb`. The `BedrockGuardrails` statement is required for contextual grounding checks deployed by `arch-review deploy`.

> **Note**: Gateway creation uses the `bedrock-agentcore-starter-toolkit` SDK, which creates a Cognito User Pool for OAuth authorization. Depending on your account setup, you may also need `cognito-idp:*` permissions for the initial `deploy` and `destroy` commands.

## Review Phases

1. **Requirements Analysis**: Extracts requirements, constraints, and NFRs from documents
2. **Architecture Analysis**: Analyzes CloudFormation templates, diagrams, and source code (queries WAF KB if available)
   - **2b. Service defaults verification**: A focused model call checks whether flagged "Features Not Found" are actually provided by AWS service defaults, filtering false positives
3. **Clarifying Questions**: Gathers context by asking the user about unverified gaps
4. **Sparring**: Triages gaps by severity, then spars on each individually with bounded follow-up rounds. Each gap is classified as CONFIRMED_GAP, ACCEPTED_RISK, or RESOLVED
5. **Final Review**: Produces structured review with gaps, risks, recommendations, and verdict

Between each phase, a **context condenser** extracts structured findings from the raw agent output using a separate model call. This prevents token overflow as context accumulates across phases. Optionally, a **contextual grounding check** validates that the condensed findings are faithful to the raw output using the Bedrock Guardrails ApplyGuardrail API.

## Input Formats

### Documents

Markdown files with requirements, constraints, NFRs, ADRs. No specific format required.

### Templates

- CloudFormation: `.yaml`, `.yml`, `.json`
- CDK: Point to `cdk.out/` directory

### Diagrams

- PNG, JPEG images
- Export draw.io files to PNG/JPEG first

## Project Structure

```
arch_sparring_agent/
├── agents/
│   ├── requirements_agent.py  # Phase 1: Document analysis
│   ├── architecture_agent.py  # Phase 2: Template/diagram/source analysis
│   ├── question_agent.py      # Phase 3: Interactive questions
│   ├── sparring_agent.py      # Phase 4: Interactive sparring
│   ├── review_agent.py        # Phase 5: Final review
│   ├── remediation_agent.py   # Remediation mode discussions
│   └── kb_tool.py             # WAF Knowledge Base query tool factory
├── cli/
│   ├── common.py              # Shared options, constants, and helpers
│   ├── run.py                 # run command
│   ├── deploy.py              # deploy/destroy commands
│   ├── remediate.py           # remediate command
│   ├── profiles.py            # profiles command group
│   └── kb.py                  # kb sync command
├── config/
│   ├── models.py              # Model registry and agent name constants
│   └── tuning.py              # Env-var-overridable tuning constants
├── infra/
│   ├── shared_config.py       # SSM-based config discovery
│   ├── gateway.py             # Gateway setup and lifecycle
│   ├── policy.py              # Cedar policy management
│   ├── guardrails.py          # Bedrock Guardrail setup and teardown
│   ├── polling.py             # Shared polling and idempotent-create utilities
│   └── memory.py              # AgentCore memory for sessions
├── kb/
│   ├── infra.py               # KB infrastructure (S3 Vectors, Bedrock KB)
│   ├── scraper.py             # WAF documentation scraper
│   └── sync.py                # S3 upload and ingestion trigger
├── review/
│   ├── orchestrator.py        # Phase orchestration + service default verification
│   ├── sparring.py            # Per-gap sparring loop (Phase 4 orchestration)
│   ├── context_condenser.py   # Structured extraction to prevent token overflow
│   ├── extraction.py          # Markdown parsing and state extraction
│   └── grounding.py           # Contextual grounding checks via Bedrock Guardrails
├── profiles/
│   ├── default.yaml           # Balanced review profile
│   ├── strict.yaml            # Strict review profile
│   └── lightweight.yaml       # Lightweight review profile
├── tools/
│   ├── common.py              # Path validation, file size checks, content search
│   ├── document_parser.py     # Markdown file reader
│   ├── cfn_analyzer.py        # CloudFormation template reader
│   ├── diagram_analyzer.py    # Diagram analysis via Bedrock
│   ├── source_analyzer.py     # Lambda/application source code reader
│   └── kb_client.py           # Knowledge Base query client
├── profiles.py                # Profile loading and resolution
├── state.py                   # Review state persistence
└── exceptions.py              # Custom exception hierarchy
```

## Development

```bash
uv sync                              # Install dependencies
uv run ruff format .                 # Format code
uv run ruff check .                  # Lint code
uv run pytest tests/ -v              # Run tests
uv run mypy arch_sparring_agent/     # Type check
```

The project uses [Hatch](https://hatch.pypa.io/) as build backend and targets Python 3.11+.

## Policy Engine

The tool automatically creates and configures a full policy enforcement stack for security:

1. **Creates a Gateway** ("ArchReviewGateway") or uses an existing one
2. **Creates a Policy Engine** ("ArchReviewPolicyEngine") or uses an existing one
3. **Creates Cedar policies** restricting each agent to specific tools:
   - **RequirementsAnalyst**: `read_document`, `list_available_documents`, `ask_user_question`
   - **ArchitectureEvaluator**: CFN/diagram/source tools (`read_cloudformation_template`, `list_cloudformation_templates`, `read_architecture_diagram`, `list_architecture_diagrams`, `list_source_files`, `read_source_file`, `search_source_code`), `query_waf`, `ask_user_question`
   - **ReviewAgent**: `query_waf`
   - **DefaultDeny**: Forbids access unless agent name matches one of the five registered agents
4. **Associates the Gateway with the Policy Engine** in `ENFORCE` mode

The QuestionAgent and SparringAgent have no tool-specific restrictions but are whitelisted in the DefaultDeny policy.

## Technical Details

- **Default Model**: Nova 2 Lite (1M context, multimodal)
- **Multi-model**: Curated registry with Nova 2 Lite and Claude Opus 4.6 via `--model`
- **Framework**: [Strands Agents SDK](https://strandsagents.com/latest/documentation/docs/)
- **Region**: eu-central-1 (configurable via `--region` or `AWS_REGION`)
- **Gateway**: AgentCore Gateway with Cognito OAuth authorization
- **Policy Engine**: AgentCore Policy Engine with Cedar policies for tool access control
- **Guardrails**: Bedrock Guardrails with contextual grounding policy for hallucination detection
- **Memory**: AgentCore Memory for remediation session persistence
- **Vector Store**: S3 Vectors (for KB)
- **Embeddings**: Amazon Titan Embed Text v2 (1024 dimensions)
- **Tracing**: Optional `[otel]` extra enables Strands SDK's built-in OpenTelemetry tracing for agent invocations and tool calls (no custom instrumentation — configure via standard OTel environment variables)

## References

- [Strands SDK](https://strandsagents.com/latest/documentation/docs/)
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Amazon Nova 2 Models](https://docs.aws.amazon.com/nova/latest/nova2-userguide/)
- [AWS Well-Architected Framework](https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html)
