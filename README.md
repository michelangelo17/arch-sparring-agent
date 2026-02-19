# Architecture Review Sparring Partner

Multi-agent system for architecture reviews. Analyzes requirements documents, CloudFormation templates, architecture diagrams, and source code, then challenges architectural decisions through interactive sparring.

## Features

- **5-phase review process**: Requirements → Architecture → Questions → Sparring → Final Review
- **Interactive sparring**: Challenges architectural gaps and pushes back on weak justifications
- **Remediation mode**: Discuss and resolve findings from previous reviews with session memory
- **CI/CD mode**: Non-interactive automated reviews with structured output and exit codes
- **CDK support**: Works with CloudFormation templates and CDK synthesized output (`cdk.out/`)
- **Multimodal analysis**: Analyzes architecture diagrams (PNG, JPEG) via Bedrock
- **Full session export**: Saves complete review session to markdown or JSON
- **Review profiles**: Customizable behavioral profiles (strict, lightweight, or your own)
- **WAF Knowledge Base**: Optional RAG-powered retrieval of AWS Well-Architected Framework best practices
- **Shared infrastructure**: Deploy once per AWS account, shared across team members

## Prerequisites

- Python 3.11+
- AWS credentials configured
- Nova 2 Lite model access in Bedrock console

## Installation

```bash
pip install arch-sparring-agent
```

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

Deploy shared infrastructure to an AWS account. Creates the Gateway, Policy Engine, and Cedar policies. Stores resource IDs in SSM Parameter Store so `arch-review run` discovers them automatically.

```bash
arch-review deploy
arch-review deploy --with-kb        # Also provision a WAF Knowledge Base
arch-review deploy --region us-east-1
```

Idempotent — safe to run repeatedly.

### `arch-review destroy`

Tear down all shared infrastructure including Gateway, Policy Engine, Knowledge Base (if present), and SSM parameter.

```bash
arch-review destroy --confirm
```

### `arch-review run`

Run an architecture review. Supports interactive and CI modes.

```bash
# Interactive (default)
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

# CI/CD mode (non-interactive)
arch-review run --ci \
    --documents-dir ./docs \
    --templates-dir ./cdk.out

# JSON output for programmatic processing
arch-review run --json \
    --documents-dir ./docs \
    --templates-dir ./templates

# Use a different profile
arch-review run --profile strict \
    --documents-dir ./docs \
    --templates-dir ./templates
```

### `arch-review remediate`

Discuss and resolve findings from a previous review:

```bash
arch-review remediate
arch-review remediate --profile lightweight
```

- Loads gaps/risks from `.arch-review/state.json`
- Continues conversations across sessions via memory
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
| `default`     | Balanced review — thorough but pragmatic                          |
| `strict`      | Low tolerance for gaps, demands evidence, errs on the side of flagging |
| `lightweight` | Pragmatic for prototypes and demos, accepts "it's a prototype"    |

```bash
arch-review run --profile strict --documents-dir ./docs --templates-dir ./templates
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

Each profile is a complete, standalone specification — no layering or overrides. Edit the generated YAML to adjust behavioral directives for each agent.

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
| `--no-history`            | Don't archive previous reviews (default in CI mode)       |
| `--keep-history`          | Archive previous reviews even in CI mode                  |
| `--no-state`              | Don't save state file after review                        |
| `--ci`                    | CI/CD mode: non-interactive analysis                      |
| `--json`                  | Output as JSON (implies --ci)                             |
| `--strict`                | Fail on any High impact risk (ignores verdict)            |
| `--reasoning-level`       | Reasoning effort: off, low, medium, high (default: low)   |
| `--skip-policy-check`     | Skip policy engine setup (development only)               |
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
| `CI`                          | Enable CI mode (true/1/yes)              |

## Exit Codes

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| 0    | PASS - no significant issues found               |
| 1    | FAIL (or --strict with High impact)              |
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

### CI/CD Environments

Each platform has its own credential mechanism:

| Platform      | Recommended Method          | Credential Source                |
| ------------- | --------------------------- | -------------------------------- |
| GitHub        | OIDC                        | IAM Identity Provider for GitHub |
| GitLab        | OIDC                        | IAM Identity Provider for GitLab |
| AWS CodeBuild | Service Role                | Attached IAM role (automatic)    |
| Jenkins       | Instance Profile or Secrets | EC2 role or credential plugin    |

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
      "Sid": "AgentCorePolicyAndGateway",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreatePolicyEngine",
        "bedrock-agentcore:ListPolicyEngines",
        "bedrock-agentcore:CreatePolicy",
        "bedrock-agentcore:UpdatePolicy",
        "bedrock-agentcore:GetPolicy",
        "bedrock-agentcore:ListPolicies",
        "bedrock-agentcore:CreateGateway",
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
        "ssm:PutParameter"
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

The `KnowledgeBaseOptional` statement is only needed if you use `deploy --with-kb`.

## CI/CD Integration

Example configurations are in `examples/ci/`.

### GitHub Actions

Copy `examples/ci/github-actions.yml` to `.github/workflows/arch-review.yml`:

- Supports OIDC (recommended) or static credentials
- Comments results on PRs
- Uploads review as artifact

### GitLab CI

Copy `examples/ci/gitlab-ci.yml` to `.gitlab-ci.yml`:

- Supports OIDC or CI/CD variables
- Runs on merge requests
- Optional JSON output job

### AWS CodeBuild

Copy `examples/ci/aws-codebuild.yml` to `buildspec.yml`:

- Uses CodeBuild service role automatically
- No credential configuration needed
- Works with CodePipeline

## Review Phases

1. **Requirements Analysis**: Extracts requirements, constraints, and NFRs from documents
2. **Architecture Analysis**: Analyzes CloudFormation templates and diagrams (queries WAF KB if available)
3. **Service Default Verification**: Filters false positives from features that AWS provides by default
4. **Clarifying Questions** (interactive) / **Gap Identification** (CI): Gathers context or identifies unknowns
5. **Sparring** (interactive) / **Risk Analysis** (CI): Challenges decisions or lists risks
6. **Final Review**: Produces structured review with gaps, risks, recommendations

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
│   ├── architecture_agent.py  # Phase 2: Template/diagram analysis
│   ├── question_agent.py      # Phase 3: Interactive questions
│   ├── sparring_agent.py      # Phase 4: Interactive sparring
│   ├── ci_agents.py           # Phase 3-4: CI/CD non-interactive
│   ├── review_agent.py        # Phase 5: Final review
│   └── remediation_agent.py   # Remediation mode discussions
├── kb/
│   ├── infra.py               # KB infrastructure (S3 Vectors, Bedrock KB)
│   ├── scraper.py             # WAF documentation scraper
│   └── sync.py                # S3 upload and ingestion trigger
├── profiles/
│   ├── default.yaml           # Balanced review profile
│   ├── strict.yaml            # Strict review profile
│   └── lightweight.yaml       # Lightweight review profile
├── tools/
│   ├── document_parser.py     # Markdown file reader
│   ├── cfn_analyzer.py        # CloudFormation template reader
│   ├── diagram_analyzer.py    # Diagram analysis via Bedrock
│   ├── source_analyzer.py     # Lambda/application source code reader
│   └── kb_client.py           # Knowledge Base query client
├── orchestrator.py            # Phase orchestration + service default verification
├── profiles.py                # Profile loading and resolution
├── infra.py                   # SharedConfig and SSM parameter management
├── config.py                  # Constants and AWS client setup
├── policy.py                  # Cedar policy management
├── gateway.py                 # Gateway setup and lifecycle
├── memory.py                  # AgentCore memory for sessions
├── exceptions.py              # Custom exception hierarchy
├── context_condenser.py       # Handles large inputs
├── state.py                   # Review state persistence
└── cli.py                     # CLI entry point (deploy/run/remediate/destroy)
examples/ci/
├── github-actions.yml         # GitHub Actions example
├── gitlab-ci.yml              # GitLab CI example
└── aws-codebuild.yml          # AWS CodeBuild example
```

## Development

```bash
uv sync                    # Install dependencies
uv run ruff format .       # Format code
uv run ruff check .        # Lint code
uv run pytest tests/ -v    # Run tests
```

## Policy Engine

The tool automatically creates and configures a full policy enforcement stack for security:

1. **Creates a Gateway** ("ArchReviewGateway") or uses an existing one
2. **Creates a Policy Engine** ("ArchReviewPolicyEngine") or uses an existing one
3. **Creates Cedar policies** restricting each agent to specific tools:
   - **RequirementsAnalyst**: Only document reading tools
   - **ArchitectureEvaluator**: Only CFN/diagram reading tools + WAF KB query
   - **ReviewAgent**: WAF KB query
   - **DefaultDeny**: Blocks unknown agents
4. **Associates the Gateway with the Policy Engine** for enforcement

## Technical Details

- **Default Model**: Nova 2 Lite (1M context, multimodal)
- **Multi-model**: Curated registry with Nova 2 Lite and Claude Opus 4.6 via `--model`
- **Framework**: AWS Strands SDK
- **Region**: eu-central-1 (configurable)
- **Policy Engine**: AgentCore Policy Engine for tool access control
- **Vector Store**: S3 Vectors (for KB)
- **Embeddings**: Amazon Titan Embed Text v2 (1024 dimensions)

## References

- [Strands SDK](https://strandsagents.com/latest/documentation/docs/)
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Amazon Nova 2 Models](https://docs.aws.amazon.com/nova/latest/nova2-userguide/)
- [AWS Well-Architected Framework](https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html)
