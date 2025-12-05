# Example Files

Example input files for testing the architecture review tool.

## Structure

- `documents/`: Markdown documents with requirements, constraints, and NFRs
- `templates/`: CloudFormation template example
- `diagrams/`: Placeholder for architecture diagrams (PNG/JPEG)

## Usage

Run the review tool with these example files:

```bash
uv run arch-review \
    --documents-dir ./examples/documents \
    --templates-dir ./examples/templates \
    --diagrams-dir ./examples/diagrams \
    --output review.md
```

## Adding Diagrams

Add PNG or JPEG architecture diagrams to the `diagrams/` directory. The tool will analyze them using multimodal capabilities.

