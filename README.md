# Architecture Review Sparring Partner

Multi-agent system for architecture reviews. Analyzes requirements documents, CloudFormation templates, and architecture diagrams, then challenges architectural decisions through interactive sparring.

## Features

- **5-phase review process**: Requirements → Architecture → Questions → Sparring → Final Review
- **Interactive sparring**: Challenges architectural gaps and pushes back on weak justifications
- **CDK support**: Works with CloudFormation templates and CDK synthesized output (`cdk.out/`)
- **Multimodal analysis**: Analyzes architecture diagrams (PNG, JPEG) via Bedrock
- **Full session export**: Saves complete review session to markdown

## Prerequisites

- Python 3.14+
- `uv` package manager
- AWS credentials configured
- Nova 2 Lite model access in Bedrock console

## Installation

```bash
# Install as CLI tool
cd arch-sparring-agent
uv tool install .

# Or run directly from source
uv run arch-review --help
```

## Updating

```bash
# Reinstall after code changes
cd arch-sparring-agent
uv tool uninstall arch-sparring-agent
uv tool install .

# Or run directly from source (always uses latest)
uv run --project /path/to/arch-sparring-agent arch-review --help
```

## Usage

```bash
arch-review \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams \
    -o review.md
```

### Options

| Option            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `--documents-dir` | Directory with markdown requirements/constraints  |
| `--templates-dir` | CloudFormation templates or `cdk.out/` directory  |
| `--diagrams-dir`  | Architecture diagrams (PNG, JPEG)                 |
| `-o, --output`    | Output file for full session                      |
| `--gateway-arn`   | Gateway ARN for policies (optional, auto-creates) |
| `--model`         | Bedrock model ID (default: Nova 2 Lite)           |
| `--region`        | AWS region (default: eu-central-1)                |

## Review Phases

1. **Requirements Analysis**: Extracts requirements, constraints, and NFRs from documents
2. **Architecture Analysis**: Analyzes CloudFormation templates and diagrams
3. **Clarifying Questions**: Gathers context about scale, security, reliability
4. **Sparring**: Challenges architectural gaps and decisions
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
│   ├── question_agent.py      # Phase 3: Clarifying questions
│   ├── sparring_agent.py      # Phase 4: Challenge decisions
│   └── review_agent.py        # Phase 5: Final review
├── tools/
│   ├── document_parser.py     # Markdown file reader
│   ├── cfn_analyzer.py        # CloudFormation template reader
│   └── diagram_analyzer.py    # Diagram analysis via Bedrock
├── orchestrator.py            # Phase orchestration
├── config.py                  # AWS/Bedrock configuration
└── cli.py                     # CLI entry point
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

To use a specific existing gateway instead of auto-creating:

```bash
arch-review \
    --gateway-arn arn:aws:bedrock-agentcore:eu-central-1:123456789012:gateway/my-gateway \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams
```

## Technical Details

- **Model**: Nova 2 Lite (300K context, multimodal)
- **Framework**: AWS Strands SDK
- **Region**: eu-central-1 (configurable)
- **Policy Engine**: AgentCore Policy Engine for tool access control (always enabled)

## References

- [Strands SDK](https://strandsagents.com/latest/documentation/)
- [AWS Bedrock Nova](https://docs.aws.amazon.com/bedrock/latest/userguide/nova.html)
