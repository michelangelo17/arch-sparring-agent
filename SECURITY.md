# Security

## Reporting Issues

If you discover a security issue, please open a [GitHub issue](https://github.com/michelangelo17/arch-sparring-agent/issues) or contact the maintainer directly.

## Important Notes

This tool interacts with your AWS account through Bedrock, AgentCore, and related services. It uses your local AWS credentials and creates resources (Gateways, Policy Engines, Cognito pools) in your account.

- **Never share** your `.arch-review/` gateway configuration files -- they contain resource identifiers.
- **Review IAM permissions** before running the tool to ensure it only has the access it needs.
- The tool enforces Cedar policies via a Policy Engine to restrict which tools each agent can use. Bypassing this with `--skip-policy-check` is intended for development only.
