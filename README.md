# autoredteam

**Find prompt injection, jailbreaks, PII leakage, and system prompt exposure in your LLM systems — automatically.**

Point it at any model or API endpoint. Get a scored, evidence-backed vulnerability report in minutes.

```
pip install glacis-autoredteam
autoredteam run --provider openai --model gpt-4o-mini
```

```
╔══════════════════════════════════════════════════════════════╗
║                    autoredteam v0.3                          ║
║         Automated Red-Teaming for AI Systems                 ║
╚══════════════════════════════════════════════════════════════╝

  Provider:  openai
  Model:     gpt-4o-mini
  Probes:    38
  Output:    results/

  ✓ 19 attack categories tested
  ✓ 4-dimension scoring (breadth · depth · novelty · reliability)
  ✓ Evidence chain with SHA-256 hash linking
  ✓ Markdown report + JSONL + attestation receipt
```

## What It Finds

autoredteam tests across **19 attack categories** using an evolutionary keep/discard loop that mutates attacks until they bypass your defenses:

| Category | Example |
|---|---|
| **Prompt injection** | Override system instructions via direct/indirect injection |
| **Jailbreaks** | Bypass safety via role-play, academic framing, fictional scenarios |
| **PII extraction** | Trick the model into leaking personal data |
| **System prompt leakage** | Extract internal instructions or system prompts |
| **Tool misuse** | Abuse available tools or trigger unintended actions |
| **Multi-turn manipulation** | Build up attacks across conversation turns |
| **Encoding bypass** | Evade filters using obfuscation or encoding |
| + 12 more | Role confusion, payload splitting, social engineering, ... |

Domain-specific attack packs are available for **healthcare**, **finance**, **HR**, and **coding agents**.

## Quickstart

```bash
# Install from PyPI
pip install glacis-autoredteam

# Dry run — echo target, no API keys needed
autoredteam run --dry-run

# Point at a real system
export OPENAI_API_KEY=sk-...
autoredteam run --provider openai --model gpt-4o-mini
```

The dry run uses an echo target that simulates a naive model. It takes about 30 seconds and shows the full loop: attack generation, scoring, evidence chain, convergence detection.

A full run against a real model takes 5–20 minutes depending on probe count and judge configuration.

Results land in `results/`:

```
results/
├── campaign_result.json       # Full structured results
├── probe_results.jsonl        # Per-probe detail
├── report.md                  # Human-readable report
└── autoharden/
    ├── policy.toml            # OVERT governance policy
    ├── hardened_prompt.txt    # Hardened system prompt
    └── evidence_chain.jsonl   # Hash-chained attestation
```

## Who It's For

- **AI/ML engineers** shipping LLM features who need to test before deploy
- **Security teams** evaluating third-party AI integrations
- **Compliance teams** needing documented evidence of adversarial testing
- **Researchers** studying LLM robustness and attack surfaces

## How It Works

autoredteam runs an evolutionary loop inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — the same keep/discard selection pattern, applied to adversarial evaluation:

```
Generate attacks (from taxonomy + mutations)
       ↓
Execute against target
       ↓
Score results (deterministic + LLM judge)
       ↓
Keep winners, discard losers
       ↓
Mutate winners, inject diversity
       ↓
Record evidence, write report
       ↓
Loop until convergence
```

**Phase 1 — Attack:** Find vulnerabilities. The loop optimizes for *higher* composite scores (more bypasses = better).

**Phase 2 — Defend:** Use winning attacks as a test suite. Harden your system prompt and guardrails. The loop now optimizes for *lower* scores (fewer bypasses = better).

**Phase 3 — Emit Policy:** Generate an [OVERT](https://overt.sh)-compliant `policy.toml` — a machine-readable governance policy any OVERT-compatible enforcement engine can consume.

### Scoring

Every attack is scored on four dimensions to prevent single-metric collapse:

| Dimension | What it measures |
|---|---|
| **Breadth** | How many attack categories find bypasses |
| **Depth** | Severity of the bypass (0 = refusal, 100 = full compliance) |
| **Novelty** | How different from prior attacks |
| **Reliability** | Does the attack reproduce consistently |

Deterministic checks run first (keyword matching, regex for PII / system prompt patterns). Ambiguous cases escalate to an LLM-as-judge with dual-judge consensus.

### Mutation Engine

Seven mutation strategies evolve attacks: rephrase, encode, nest, persona shift, language switch, format change, authority escalation. A diversity injection mechanism fires every N cycles to prevent premature convergence.

## Multi-Cloud Support

Works with any LLM provider:

```bash
autoredteam run --provider openai --model gpt-4o-mini
autoredteam run --provider anthropic --model claude-sonnet-4-5
autoredteam run --provider bedrock --model claude-sonnet-4 --region us-east-1
autoredteam run --provider google --model gemini-2.0-flash
autoredteam run --provider azure_openai --model gpt-4o
autoredteam run --provider cloudflare --model @cf/meta/llama-3-8b-instruct
```

Or bring your own target:

```python
from prepare import Target, TargetCapabilities, TARGET_REGISTRY

class MyTarget(Target):
    def send(self, prompt: str) -> str:
        return my_api.chat(prompt)

    def reset(self) -> None:
        my_api.new_session()

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True)

TARGET_REGISTRY["my_target"] = MyTarget
```

## CLI Reference

```bash
# Red-team campaigns
autoredteam run --dry-run                                     # Echo target, no API keys
autoredteam run --provider openai --model gpt-4o-mini         # Full run
autoredteam run --pack generic_taxonomy healthcare            # Multiple attack packs
autoredteam run --stealth-profile medium                      # Stealth mode
autoredteam run --judge-backend api                           # LLM-as-judge scoring

# Validation suites
autoredteam validate --suite generic --provider openai --model gpt-4o-mini

# Policy generation
autoredteam emit-policy results/autoharden/                   # Generate OVERT policy.toml

# Discovery
autoredteam providers list                                    # Available providers
autoredteam packs list                                        # Available attack packs
```

## Evidence & Attestation

autoredteam maintains a tamper-evident evidence chain using SHA-256 chain hashing.

**Free (local, default):** Every result is recorded in `evidence_chain.jsonl` with chain hashes. The chain is verifiable — `attestation.py` recomputes all hashes and detects tampering. Pass `--attest` to emit `attestation_receipt.json`.

**Paid (Glacis):** Cryptographic attestation via the Glacis service. Chain hashes are submitted for timestamping and tamper-proof storage — useful for compliance, audits, and responsible disclosure timelines.

Hash separation keeps sensitive attack details scoped:

| Tier | Contains |
|---|---|
| **Public** | Summary stats, category coverage, composite scores |
| **Team** | Full hashes, score vectors, timestamps |
| **Admin** | Raw prompts and responses (local-only, gitignored) |

## OVERT Policy Output

The autoharden loop generates an [OVERT](https://overt.sh)-compliant `policy.toml` capturing what was learned during hardening:

```bash
# Generate policy from autoharden results
autoredteam emit-policy results/autoharden/ --profile healthcare-ambient

# From a specific report
autoredteam emit-policy results/autoharden/autoharden_report.json -o deployment/policy.toml
```

The policy includes input/output filtering rules, violation types, tool-call deny rules, the hardened system prompt, and attestation config — all traceable via SHA-256 chain hash back to the evidence chain.

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
target:
  type: openai              # openai, anthropic, gemini, azure_openai,
  params:                   # bedrock, cloudflare, openai_compatible, echo
    model: gpt-4o-mini
    system_prompt: "You are a helpful customer service bot."

campaign:
  max_probes: 20
  intensity: medium         # low, medium, high
  stealth_profile: none     # none, light, medium, aggressive

scoring:
  judge_backend: deterministic  # deterministic, api, slm
  weights: { breadth: 0.25, depth: 0.25, novelty: 0.25, reliability: 0.25 }
```

## Roadmap

- [x] v0.1 — Single-turn text attacks, deterministic + LLM scoring, local evidence chain
- [x] v0.2 — Multi-turn attack chains, agentic target support
- [x] v0.3 — Autoharden self-healing loop, OVERT policy.toml output, multi-cloud providers
- [ ] v0.4 — Image/multimodal attack vectors, recursive policy hardening
- [ ] v1.0 — Full OVERT standard conformance, compliance reporting

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All contributions welcome — from bug reports to new attack packs.

## Citation

If you use autoredteam in research, see [CITATION.cff](CITATION.cff) or cite:

```bibtex
@software{autoredteam,
  title = {autoredteam: Automated Red-Teaming for AI Systems},
  author = {Glacis},
  url = {https://github.com/glacis-io/auto-redteam},
  license = {Apache-2.0}
}
```

## License

[Apache 2.0](LICENSE)

---

Built by [Glacis](https://glacis.io). The open-source tool is free forever. Cryptographic attestation is the paid upgrade.
