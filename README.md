# Architecture Review Sparring Partner

Multi-agent system for comprehensive architecture reviews using AWS Bedrock and Strands SDK. Analyzes requirements documents, CloudFormation templates (including CDK synthesized output), and architecture diagrams to provide structured reviews.

## Features

- **Multi-agent architecture**: Three specialized agents working together
  - Requirements Analyst: Extracts and structures requirements from documents
  - Architecture Evaluator: Analyzes CloudFormation templates and diagrams
  - Review Moderator: Synthesizes findings into comprehensive reviews
- **AgentCore integration**: Memory, policy controls, and quality evaluations
- **CDK support**: Works with both regular CloudFormation templates and CDK synthesized output
- **Multimodal analysis**: Analyzes architecture diagrams using Bedrock's multimodal capabilities
- **Interactive questioning**: Agents can ask clarifying questions when information is missing

## Prerequisites

- Python 3.14+
- `uv` package manager
- AWS credentials configured (`~/.aws/credentials` or environment variables)
- Access to Nova 2 Lite model in Bedrock console (request model access)

## Installation

### Option 1: Run from Source (Development)

```bash
# Clone or navigate to the project directory
cd arch-sparring-agent

# Install dependencies
uv sync

# Verify installation
uv run arch-review --help
```

### Option 2: Install as Package

```bash
# From the project directory
uv pip install -e .

# Or install from a git repository
uv pip install git+https://github.com/yourusername/arch-sparring-agent.git
```

## Quick Start

1. **Prepare input directories**:

   ```bash
   mkdir -p my-project/{documents,templates,diagrams}
   ```

2. **Add your files**:

   - `documents/`: Markdown files with requirements, constraints, NFRs
   - `templates/`: CloudFormation templates (`.yaml`, `.yml`, `.json`) or CDK `cdk.out/` directory
   - `diagrams/`: Architecture diagrams (PNG, JPEG images)

3. **Run the review**:

   ```bash
   uv run arch-review \
       --documents-dir ./my-project/documents \
       --templates-dir ./my-project/templates \
       --diagrams-dir ./my-project/diagrams \
       --output review.md
   ```

4. **Test with examples**:
   ```bash
   uv run arch-review \
       --documents-dir ./examples/documents \
       --templates-dir ./examples/templates \
       --diagrams-dir ./examples/diagrams \
       --output review.md
   ```

## Usage

### Basic Usage

```bash
# With regular CloudFormation templates
uv run arch-review \
    --documents-dir ./docs \
    --templates-dir ./templates \
    --diagrams-dir ./diagrams \
    --output review.md

# With CDK synthesized output
uv run arch-review \
    --documents-dir ./docs \
    --templates-dir ./cdk.out \
    --diagrams-dir ./diagrams \
    --output review.md
```

### Command Line Options

- `--documents-dir`: Directory containing markdown documents (problem statement, goals, NFRs, ADRs)
- `--templates-dir`: Directory containing CloudFormation templates or CDK `cdk.out/` directory
- `--diagrams-dir`: Directory containing architecture diagrams (PNG, JPEG images)
- `--output`: Output file path for the review (optional, prints to stdout if not specified)
- `--model`: Bedrock model ID (default: `amazon.nova-2-lite-v1:0` - Nova 2 Lite)
- `--region`: AWS region (default: `eu-central-1`)
- `--interactive`: Enable interactive questioning mode

## Input Format

### Documents Directory

Markdown files containing:

- Problem statements
- Goals and objectives
- Constraints
- Non-functional requirements (NFRs)
- Architectural Decision Records (ADRs)

Documents can be in any format - agents extract relevant information semantically.

### Templates Directory

- Regular CloudFormation templates (`.yaml`, `.yml`, `.json`)
- CDK synthesized output directory (`cdk.out/`) containing `.template.json` files

Note: CDK synthesized output is CloudFormation, so the same analyzer handles both.

### Diagrams Directory

Architecture diagrams as images:

- Supported formats: PNG, JPEG
- For draw.io files: Export to PNG or JPEG first

## How It Works

1. **Requirements Agent** analyzes markdown documents to extract requirements, constraints, and goals
2. **Architecture Agent** analyzes CloudFormation templates and diagrams to understand implementation
3. **Moderator Agent** coordinates both agents and synthesizes findings into a comprehensive review

The moderator agent uses the Agents-as-Tools pattern, calling specialized agents as needed.

## AgentCore Features

### Memory

Agents retain context across sessions using AgentCore Memory. Enabled by default.

### Policy Controls

Enforce tool access restrictions and rate limiting using Cedar policies:

```python
from arch_sparring_agent.config import setup_architecture_review_policies

# Set up policies for all agents
setup_architecture_review_policies()
```

### Quality Evaluations

Monitor agent performance using Online Evaluations. Configure via `setup_online_evaluation()` in `config.py`.

## Architecture

```
arch-sparring-agent/
├── arch_sparring_agent/
│   ├── agents/
│   │   ├── requirements_agent.py    # Requirements analysis
│   │   ├── architecture_agent.py     # Implementation analysis
│   │   └── moderator_agent.py       # Review coordination
│   ├── tools/
│   │   ├── document_parser.py       # Read markdown documents
│   │   ├── cfn_analyzer.py          # Read CloudFormation templates
│   │   └── diagram_analyzer.py      # Analyze architecture diagrams
│   ├── config.py                    # Bedrock and AgentCore configuration
│   ├── orchestrator.py              # Main orchestration logic
│   └── cli.py                       # CLI entry point
└── pyproject.toml
```

## Technical Details

- **Language**: Python 3.14+
- **Package Manager**: `uv`
- **Agent Framework**: AWS Strands SDK
- **Bedrock Model**: Nova 2 Lite (`amazon.nova-2-lite-v1:0`)
  - 300K token context window
  - Multimodal support (text, images, videos)
- **Region**: eu-central-1 (Frankfurt)
- **Agent Pattern**: Agents-as-Tools (hierarchical delegation)

## Development

```bash
# Install development dependencies
uv sync

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## References

- [Strands SDK Documentation](https://strandsagents.com/latest/documentation/)
- [AWS Bedrock Nova Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/nova.html)
- [AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [Cedar Policy Language](https://www.cedarpolicy.com/)
