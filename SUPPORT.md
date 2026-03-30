# Support

## Getting Help

- **Bug reports:** [Open an issue](https://github.com/glacis-io/auto-redteam/issues/new?template=bug_report.yml)
- **Feature requests:** [Open an issue](https://github.com/glacis-io/auto-redteam/issues/new?template=feature_request.yml)
- **Questions & discussion:** [GitHub Discussions](https://github.com/glacis-io/auto-redteam/discussions)
- **Security issues:** See [SECURITY.md](SECURITY.md) — do not open a public issue

## Quick Troubleshooting

**"No API key" errors:**
```bash
export OPENAI_API_KEY=sk-...        # For OpenAI targets
export ANTHROPIC_API_KEY=sk-ant-... # For Anthropic targets
```

**Test without API keys:**
```bash
autoredteam run --dry-run
```

**Provider not found:**
```bash
autoredteam providers list          # See available providers
pip install "glacis-autoredteam[openai]"  # Install provider extras
```

## Enterprise Support

For enterprise features including cryptographic attestation, compliance reporting, and dedicated support, contact **eng@glacis.io** or visit [glacis.io](https://glacis.io).
