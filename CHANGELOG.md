# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-08

### Added

- Initial release
- 5-phase architecture review process:
  - Phase 1: Requirements analysis from markdown documents
  - Phase 2: Architecture analysis from CloudFormation templates and diagrams
  - Phase 3: Gap identification (CI) / Clarifying questions (interactive)
  - Phase 4: Risk analysis (CI) / Architecture sparring (interactive)
  - Phase 5: Final review with verdict
- Source code analysis via `--source-dir` option
- CI/CD mode with non-interactive operation (`--ci` flag)
- JSON output for programmatic processing (`--json` flag)
- Strict mode for failing on any High impact risk (`--strict` flag)
- Automatic Gateway and Policy Engine setup
- Cedar policies for agent tool access control
- AgentCore memory for session context
- Support for multimodal analysis (PNG/JPEG diagrams)
- Example CI configurations for GitHub Actions, GitLab CI, AWS CodeBuild

### Security

- Policy Engine enforces agent tool restrictions
- Default deny policy for unknown agents
- OAuth authorization via Cognito

