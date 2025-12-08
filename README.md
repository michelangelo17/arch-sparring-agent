# Architecture Review Sparring Partner

Multi-agent system for architecture reviews. Analyzes requirements documents, CloudFormation templates, architecture diagrams, and source code, then challenges architectural decisions through interactive sparring.

## Features

- **5-phase review process**: Requirements → Architecture → Questions → Sparring → Final Review
- **Interactive sparring**: Challenges architectural gaps and pushes back on weak justifications
- **CI/CD mode**: Non-interactive automated reviews with structured output and exit codes
- **CDK support**: Works with CloudFormation templates and CDK synthesized output (`cdk.out/`)
- **Multimodal analysis**: Analyzes architecture diagrams (PNG, JPEG) via Bedrock
- **Full session export**: Saves complete review session to markdown or JSON

## Prerequisites

- Python 3.11+
- AWS credentials configured
- Nova 2 Lite model access in Bedrock console

## Installation

```bash
pip install arch-sparring-agent
```

## Usage

### Interactive Mode (Default)

```bash
arch-review \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams \
    -o review.md
```

### With Source Code Analysis

CloudFormation templates only define infrastructure. To analyze business logic in Lambda handlers:

```bash
arch-review \
    --documents-dir ./docs \
    --templates-dir ./cdk.out \
    --diagrams-dir ./diagrams \
    --source-dir ./src/lambdas \
    -o review.md
```

### CI/CD Mode

```bash
# Non-interactive with markdown output
arch-review --ci \
    --documents-dir ./docs \
    --templates-dir ./cdk.out \
    --diagrams-dir ./diagrams \
    --source-dir ./src \
    -o review.md

# JSON output for programmatic processing
arch-review --json \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams
```

### Options

| Option            | Description                                      |
| ----------------- | ------------------------------------------------ |
| `--documents-dir` | Directory with markdown requirements/constraints |
| `--templates-dir` | CloudFormation templates or `cdk.out/` directory |
| `--diagrams-dir`  | Architecture diagrams (PNG, JPEG)                |
| `--source-dir`    | Lambda/application source code (optional)        |
| `-o, --output`    | Output file for full session                     |
| `--ci`            | CI/CD mode: non-interactive analysis             |
| `--json`          | Output as JSON (implies --ci)                    |
| `--strict`        | Fail on any High impact risk (ignores verdict)   |
| `--model`         | Bedrock model ID (default: Nova 2 Lite)          |
| `--region`        | AWS region (default: eu-central-1)               |

### Environment Variables

All options can be set via environment variables:

| Variable                    | Description                 |
| --------------------------- | --------------------------- |
| `ARCH_REVIEW_DOCUMENTS_DIR` | Documents directory         |
| `ARCH_REVIEW_TEMPLATES_DIR` | Templates directory         |
| `ARCH_REVIEW_DIAGRAMS_DIR`  | Diagrams directory          |
| `ARCH_REVIEW_SOURCE_DIR`    | Source code directory       |
| `ARCH_REVIEW_OUTPUT`        | Output file path            |
| `ARCH_REVIEW_MODEL`         | Bedrock model ID            |
| `AWS_REGION`                | AWS region                  |
| `CI`                        | Enable CI mode (true/1/yes) |

### Exit Codes

| Code | Meaning                             |
| ---- | ----------------------------------- |
| 0    | PASS or PASS WITH CONCERNS          |
| 1    | FAIL (or --strict with High impact) |
| 3    | Error during execution              |

## AWS Credentials

The tool uses the standard AWS credential chain. No credentials are hardcoded.

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

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:ListFoundationModels",
        "bedrock-agentcore:*",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

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
2. **Architecture Analysis**: Analyzes CloudFormation templates and diagrams
3. **Clarifying Questions** (interactive) / **Gap Identification** (CI): Gathers context or identifies unknowns
4. **Sparring** (interactive) / **Risk Analysis** (CI): Challenges decisions or lists risks
5. **Final Review**: Produces structured review with gaps, risks, recommendations

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
│   └── review_agent.py        # Phase 5: Final review
├── tools/
│   ├── document_parser.py     # Markdown file reader
│   ├── cfn_analyzer.py        # CloudFormation template reader
│   ├── diagram_analyzer.py    # Diagram analysis via Bedrock
│   └── source_analyzer.py     # Lambda/application source code reader
├── orchestrator.py            # Phase orchestration
├── config.py                  # AWS/Bedrock configuration
└── cli.py                     # CLI entry point
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
```

## Policy Engine

The tool automatically creates and configures a full policy enforcement stack for security:

1. **Creates a Gateway** ("ArchReviewGateway") or uses an existing one
2. **Creates a Policy Engine** ("ArchReviewPolicyEngine") or uses an existing one
3. **Creates Cedar policies** restricting each agent to specific tools:
   - **RequirementsAnalyst**: Only document reading tools
   - **ArchitectureEvaluator**: Only CFN/diagram reading tools
   - **ReviewModerator**: Only agent communication tools
   - **DefaultDeny**: Blocks unknown agents
4. **Associates the Gateway with the Policy Engine** for enforcement

## Technical Details

- **Model**: Nova 2 Lite (300K context, multimodal)
- **Framework**: AWS Strands SDK
- **Region**: eu-central-1 (configurable)
- **Policy Engine**: AgentCore Policy Engine for tool access control (always enabled)

## References

- [Strands SDK](https://strandsagents.com/latest/documentation/docs/)
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Amazon Nova 2 Models](https://docs.aws.amazon.com/nova/latest/nova2-userguide/)
