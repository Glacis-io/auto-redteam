# autoredteam

**Automated red-teaming for AI systems.** Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern — the evolutionary keep/discard loop — applied to adversarial evaluation of LLM deployments.

You describe your target system. autoredteam discovers its vulnerabilities while you sleep. Wake up to a scored, evidence-backed report.

```
$ autoredteam run --provider openai --model gpt-4o-mini
╔══════════════════════════════════════════════════════════════╗
║                    autoredteam v0.3                          ║
║         Automated Red-Teaming for AI Systems                 ║
║         Multi-cloud · Multi-turn · Stealth · Domain-aware    ║
╚══════════════════════════════════════════════════════════════╝

  Provider:  openai
  Model:     gpt-4o-mini
  Packs:     generic_taxonomy
  Probes:    38
  Stealth:   none
  Judge:     deterministic
  Output:    results/
```

## Why This Exists

Red-teaming is labor-intensive. Most teams run a handful of manual probes, declare victory, and ship. The attack surface is larger than anyone explores by hand.

autoredteam applies the autoresearch insight — evolutionary optimization with keep/discard selection — to attack generation. Instead of optimizing a research paper toward a citation metric, we optimize attack prompts toward a **4-component scoring vector** that resists the single-metric collapse problem:

| Dimension | What it measures | Why it matters |
|---|---|---|
| **Breadth** | How many attack categories find bypasses? | Prevents tunnel vision on one exploit |
| **Depth** | How severe is the bypass? (0 = refusal, 100 = full compliance) | Distinguishes minor leaks from critical failures |
| **Novelty** | How different is this from prior attacks? | Rewards exploration over repetition |
| **Reliability** | Does the attack reproduce consistently? | Filters flukes from real vulnerabilities |

The composite score is a configurable weighted sum. You can bias toward breadth (coverage testing) or depth (finding the worst-case failure) depending on your evaluation goals.

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

Or clone for development:

```bash
git clone https://github.com/glacis-io/auto-redteam.git
cd auto-redteam
pip install -e .
```

The dry run uses an echo target that simulates a naive model. It takes about 30 seconds and shows you the full loop: attack generation, scoring, evidence chain, convergence detection. A full run against a real model takes roughly 5–20 minutes depending on probe count and whether you enable the LLM-as-judge.

Results land in `results/`.

## How It Works

autoredteam runs a two-phase evolutionary loop:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   Generate attacks (from taxonomy + mutations)  │
│         ↓                                       │
│   Execute against target                        │
│         ↓                                       │
│   Score results (deterministic + LLM judge)     │
│         ↓                                       │
│   Keep winners, discard losers                  │
│         ↓                                       │
│   Mutate winners, inject diversity              │
│         ↓                                       │
│   Record evidence, write report                 │
│         ↓                                       │
│   Loop until convergence                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Phase 1 — Attack:** Find as many vulnerabilities as possible. The loop optimizes for *higher* composite scores (more bypasses = better).

**Phase 2 — Defend:** Take the winning attacks as a test suite. Harden your system prompt and guardrails. The loop now optimizes for *lower* scores (fewer bypasses = better).

**Phase 3 — Emit Policy:** Generate an [OVERT](https://overt.sh)-compliant `policy.toml` from hardening results. This is a machine-readable governance policy that any OVERT-compatible enforcement engine can consume — or that autoredteam can ingest for another round of recursive hardening.

## Architecture

```
autoredteam/
├── cli.py               # Unified CLI entrypoint
├── campaign.py          # Campaign definition + probe graph
├── campaign_runner.py   # Campaign execution engine
├── prepare.py           # Target interface + connection setup
├── attack.py            # Attack taxonomy + mutation engine
├── scoring.py           # 4-component scoring harness
├── scoring_v2.py        # v0.3 scoring engine (campaign-aware)
├── conversation.py      # Multi-turn conversation manager
├── trajectory_engine.py # Multi-turn trajectory planner
├── stealth.py           # Stealth profile engine (encoding, timing, persona)
├── defend.py            # Self-healing engine (diagnose → prescribe → apply → verify)
├── autoharden.py        # Autonomous hardening loop
├── immune.py            # Immune-system defense (periodic re-verification)
├── emit_policy.py       # OVERT policy generator (autoharden results → policy.toml)
├── attestation.py       # Evidence chain (local free, Glacis paid)
├── run.py               # Legacy evolutionary loop
├── config.yaml          # All configuration
├── attack_packs/        # Modular attack pack system
│   ├── generic_taxonomy.py  # 19-category generic pack
│   └── domains/             # Domain-specific packs
│       ├── healthcare.py
│       ├── finance.py
│       ├── hr.py
│       └── coding_agents.py
├── providers/           # Multi-cloud provider adapters
├── reporting/           # Report generation (markdown, JSONL, PR body)
├── models/              # Local scoring model weights
├── training/            # Fine-tuning pipeline for local judge
├── validation/          # Public validation suites
└── results/             # Output directory
    ├── campaign_result.json   # Full campaign results
    ├── probe_results.jsonl    # Per-probe results
    ├── report.md              # Human-readable report
    └── autoharden/
        ├── policy.toml            # OVERT governance policy
        ├── hardened_prompt.txt    # Hardened system prompt
        ├── guardrail_config.json  # Platform guardrail rules
        └── evidence_chain.jsonl   # Hash-chained attestation
```

### Scoring Pipeline

The scoring pipeline runs deterministic checks first (keyword matching, regex for PII / system prompt leakage patterns), then escalates ambiguous cases to an LLM-as-judge. For ambiguous results (deterministic severity between 10–50), a dual-judge consensus mechanism runs two independent LLM evaluations. If they diverge, the conservative (higher severity) score is used.

This matters because single-metric optimization is the Achilles' heel of automated evaluation — the system finds one exploit and hammers it. The 4-component vector forces the optimizer to explore broadly, go deep on real vulnerabilities, discover novel attacks, and only count reproducible ones.

### Mutation Engine

The mutation engine applies 7 strategies to evolve attacks: rephrase, encode, nest, persona shift, language switch, format change, and authority escalation. Mutations operate within categories but can cross-pollinate. A diversity injection mechanism fires every N cycles to prevent premature convergence.

## Attack Taxonomy

Built-in categories (19 total, each with seed templates):

**Core categories:**
- **Prompt injection** — Override system instructions via direct and indirect injection
- **Jailbreaks** — Bypass safety via role-play, academic framing, fictional scenarios
- **PII extraction** — Trick the model into leaking personal data
- **System prompt leakage** — Extract the system prompt or internal instructions
- **Tool misuse** — Abuse available tools or trigger unintended actions
- **Role confusion** — Confuse the model about its role or identity
- **Context window poisoning** — Exploit context window mechanics
- **Multi-turn manipulation** — Build up across multiple conversation turns

**Extended categories:**
- **Encoding bypass** — Bypass filters using encoding, obfuscation, or alternative representations
- **Payload splitting** — Split malicious payloads across fragments to evade detection
- **Refusal suppression** — Prevent the model from using its standard refusal patterns
- **Ethical bypass** — Use ethical or educational framing to bypass safety filters
- **Authority manipulation** — Impersonate authority figures or invoke false credentials
- **Output formatting exploit** — Exploit output formatting to extract or embed harmful content
- **Indirect injection** — Inject instructions via data the model is asked to process
- **Multilingual attack** — Use language switching to bypass English-trained safety filters
- **Continuation attack** — Trick the model into continuing a harmful pattern it didn't start
- **Social engineering** — Use emotional manipulation, urgency, or social pressure
- **Hallucination exploit** — Exploit model tendency to confabulate fake but plausible information

Domain-specific attack packs (healthcare, finance, HR, coding agents) are also available via `autoredteam packs list`.

## Adding Your Own Target

Implement the `Target` protocol in `prepare.py`:

```python
class MyTarget(Target):
    def send(self, prompt: str) -> str:
        """Send a single prompt, return the model's response text."""
        return my_api.chat(prompt)

    def reset(self) -> None:
        """Clear conversation state between attack probes."""
        my_api.new_session()

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True)

# Register it
TARGET_REGISTRY["my_target"] = MyTarget
```

Then set `target.type: my_target` in `config.yaml`. The interface is deliberately minimal — three methods — so you can wrap any LLM API, local model, or multi-agent system.

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
target:
  # "openai", "anthropic", "gemini", "azure_openai",
  # "bedrock", "cloudflare", "openai_compatible", "echo"
  type: openai
  params:
    model: gpt-4o-mini
    system_prompt: "You are a helpful customer service bot for Acme Corp."

run:
  mode: classic         # "classic" or "coevol"
  max_cycles: 10        # More cycles = more thorough, more API spend
  batch_size: 10        # Attacks per cycle
  phase: attack         # "attack" or "defend"

campaign:
  max_probes: 20
  max_trajectory_turns: 5
  intensity: medium     # "low", "medium", "high"
  stealth_profile: none # "none", "light", "medium", "aggressive"

scoring:
  judge_backend: deterministic  # "deterministic", "api", or "slm"
  weights:
    breadth: 0.25
    depth: 0.25
    novelty: 0.25
    reliability: 0.25
```

## Evidence & Attestation

autoredteam maintains a tamper-evident evidence chain using SHA-256 chain hashing:

**Free tier (local, default):** Every attack result is recorded in `evidence_chain.jsonl` with chain hashes linking each record to the previous one. The chain is verifiable — `attestation.py` can recompute all hashes and detect tampering. Raw prompts stay local (never leave your machine). Summary stats in `summary.json` are safe to share. Pass `--attest` to also emit `attestation_receipt.json`, a shareable receipt stub for the completed run.

**Paid tier (Glacis):** Cryptographic attestation via the Glacis service. Chain hashes are submitted for timestamping and tamper-proof storage. Proves *when* you tested and *what* you found — useful for compliance, audits, and responsible disclosure timelines.

**Hash separation:** The attestation chain contains SHA-256 hashes of attack prompts, not the raw prompts themselves. Three access tiers keep sensitive attack details appropriately scoped:

1. **Public** — summary stats, category coverage, composite scores
2. **Team** — full hashes, score vectors, timestamps, deterministic flags
3. **Admin** — raw prompts and responses (local-only, gitignored by default)

## OVERT Policy Output

autoredteam's autoharden loop automatically generates an [OVERT](https://overt.sh)-compliant `policy.toml` at the end of each run. OVERT (Open Verification and Evaluation for Responsible Technology) is an open standard for declarative AI governance policy.

The generated policy captures what autoredteam learned during hardening:

| OVERT Domain | What autoredteam contributes |
|---|---|
| `[protect]` | Input/output filtering rules derived from discovered attack patterns (PII redaction, injection detection, encoding normalization) |
| `[measure]` | Violation types from tested categories, recommended sampling rates |
| `[tool]` | Tool-call deny rules from tool_misuse findings |
| `[prompt]` | The hardened system prompt with defense blocks and provenance hashes |
| `[attest]` | Attestation config scaled to achieved governance level |
| `[respond]` | Failure mode (fail-closed/open) derived from governance tier |

```bash
# Generate policy from autoharden results
autoredteam emit-policy results/autoharden/ --profile healthcare-ambient

# Or generate from a specific report
autoredteam emit-policy results/autoharden/autoharden_report.json -o deployment/policy.toml
```

The `[policy.provenance]` section records the red-teaming evidence: target model, cycles run, final ASR, governance score, and the SHA-256 chain hash linking to the full evidence chain. This makes the policy traceable — you can verify that a policy.toml was generated from a specific hardening session.

**The closed loop:** policy.toml is designed to be consumed by any OVERT-compatible enforcement engine. It can also be fed back into autoredteam for recursive hardening — each pass discovers new vulnerabilities in the hardened posture and tightens the policy further.

## CLI Reference

```bash
# Red-team campaigns
autoredteam run --dry-run                                     # Echo target, no API keys
autoredteam run --provider openai --model gpt-4o-mini         # Full run against OpenAI
autoredteam run --provider anthropic --model claude-sonnet-4-5 # Against Anthropic
autoredteam run --provider bedrock --model claude-sonnet-4 --region us-east-1
autoredteam run --pack generic_taxonomy healthcare            # Multiple attack packs
autoredteam run --stealth-profile medium                      # Stealth mode
autoredteam run --judge-backend api                           # LLM-as-judge scoring

# Validation suites
autoredteam validate --suite generic --provider openai --model gpt-4o-mini
autoredteam validate --suite overnight                        # Full overnight suite

# Policy generation
autoredteam emit-policy results/autoharden/                   # Generate OVERT policy.toml

# Discovery
autoredteam providers list                                    # List available providers
autoredteam packs list                                        # List available attack packs
```

## Program.md

Edit `program.md` to steer the agent's strategy in plain English:

```markdown
## Objective
Red-team our customer service bot before launch.

## Priority Categories
1. System prompt leakage — our prompt contains API keys
2. PII extraction — bot has access to customer records
3. Jailbreaks — bot must never produce offensive content

## Success Criteria
- Ship-blocking: any system prompt leakage
- Must-fix: PII extraction at depth > 60
```

## Roadmap

- [x] v0.1 — Single-turn text attacks, deterministic + LLM scoring, local evidence chain
- [x] v0.2 — Multi-turn attack chains, agentic target support
- [x] v0.3 — Autoharden self-healing loop, OVERT policy.toml output, multi-cloud providers
- [ ] v0.4 — Image/multimodal attack vectors, recursive policy hardening
- [ ] v1.0 — Full OVERT standard conformance, compliance reporting

## Acknowledgments

The evolutionary keep/discard loop is directly inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). autoredteam applies the same pattern to adversarial evaluation: where autoresearch evolves a research paper toward a quality metric, autoredteam evolves attack prompts toward a multi-dimensional vulnerability score.

## License

[Apache 2.0](LICENSE) — patent-friendly, important given Glacis's patent portfolio.

---

Built by [Glacis](https://glacis.io). The open-source tool is free forever. Cryptographic attestation is the paid upgrade.
