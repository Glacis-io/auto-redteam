# autoredteam

**Automated red-teaming for AI systems.** Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern — the evolutionary keep/discard loop — applied to adversarial evaluation of LLM deployments.

You describe your target system. autoredteam discovers its vulnerabilities while you sleep. Wake up to a scored, evidence-backed report.

```
$ python run.py --config config.yaml
╔══════════════════════════════════════════════════════════╗
║                    autoredteam v0.1                      ║
║         Automated Red-Teaming for AI Systems             ║
╚══════════════════════════════════════════════════════════╝

⚔️  Cycle 3 — ATTACK phase
  Attacks run:    10
  Hits (bypass):  4 (40%)
  Best composite: 72.5
  Categories bypassed: system_prompt_leakage, prompt_injection, jailbreak
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
# Clone and install
git clone https://github.com/glacis-io/auto-redteam.git
cd autoredteam
uv pip install -r requirements.txt  # or: pip install -r requirements.txt

# Dry run — echo target, no API keys needed
python run.py --dry-run

# Point at a real system
export OPENAI_API_KEY=sk-...
# Edit config.yaml: set target.type to "openai" and target.params.model
python run.py --attest
```

The dry run uses an echo target that simulates a naive model. It takes about 30 seconds and shows you the full loop: attack generation, scoring, evidence chain, convergence detection. A full run against a real model with 10 cycles takes roughly 5–20 minutes depending on batch size and whether you enable the LLM-as-judge.

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

**Phase 2 — Defend:** Take the winning attacks as a test suite. Harden your system prompt and guardrails. Re-run with `--phase defend`. The loop now optimizes for *lower* scores (fewer bypasses = better).

## Architecture

```
autoredteam/
├── prepare.py       # Target interface + connection setup (READ-ONLY during loop)
├── attack.py        # Attack taxonomy + mutation engine (agent modifies this)
├── scoring.py       # 4-component scoring harness
├── attestation.py   # Evidence chain (local free, Glacis paid)
├── run.py           # Main evolutionary loop
├── program.md       # YOUR strategy in plain English
├── config.yaml      # All configuration
└── results/         # Output directory
    ├── evidence_chain.jsonl   # Tier 2: hashes + scores
    ├── summary.json           # Tier 1: public stats
    ├── attestation_receipt.json # Explicit receipt artifact (`--attest`)
    └── final_report.json      # Complete run results
```

The autoresearch pattern: `prepare.py` is read-only — it defines the target interface and never changes during a run. `attack.py` is the file the agent evolves each cycle. `program.md` is the human's strategy document that steers the loop.

### Scoring Pipeline

The scoring pipeline runs deterministic checks first (keyword matching, regex for PII / system prompt leakage patterns), then escalates ambiguous cases to an LLM-as-judge. For ambiguous results (deterministic severity between 10–50), a dual-judge consensus mechanism runs two independent LLM evaluations. If they diverge, the conservative (higher severity) score is used.

This matters because single-metric optimization is the Achilles' heel of automated evaluation — the system finds one exploit and hammers it. The 4-component vector forces the optimizer to explore broadly, go deep on real vulnerabilities, discover novel attacks, and only count reproducible ones.

### Mutation Engine

The mutation engine applies 7 strategies to evolve attacks: rephrase, encode, nest, persona shift, language switch, format change, and authority escalation. Mutations operate within categories but can cross-pollinate. A diversity injection mechanism fires every N cycles to prevent premature convergence.

## Attack Taxonomy

Built-in categories (8 total, each with seed templates):

- **Prompt injection** — Override system instructions via direct and indirect injection
- **Jailbreaks** — Bypass safety via role-play, academic framing, fictional scenarios
- **PII extraction** — Trick the model into leaking personal data
- **System prompt leakage** — Extract the system prompt or internal instructions
- **Tool misuse** — Abuse available tools or function calling
- **Role confusion** — Confuse the model about its identity or authority level
- **Context window poisoning** — Exploit attention mechanics with long prefixes
- **Multi-turn manipulation** — Build up across conversation turns (single-turn approximations in v0.1)

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
  type: openai          # "openai", "anthropic", or "echo"
  params:
    model: gpt-4o-mini
    system_prompt: "You are a helpful customer service bot for Acme Corp."

run:
  max_cycles: 10        # More cycles = more thorough, more API spend
  batch_size: 10        # Attacks per cycle
  phase: attack         # "attack" or "defend"

scoring:
  use_llm_judge: true   # Needs an OpenAI key; false = deterministic-only
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

## CLI Reference

```bash
python run.py --dry-run                          # Echo target, no API keys
python run.py --config config.yaml               # Full attack run
python run.py --config config.yaml --attest      # Full run + attestation receipt
python run.py --config config.yaml --phase defend # Defend phase
python run.py --cycles 3 --dry-run               # Quick 3-cycle test
python run.py --quiet                            # Final results only
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
- [ ] v0.2 — Multi-turn attack chains, agentic target support
- [ ] v0.3 — Tool-use attacks, function calling probes
- [ ] v0.4 — Image/multimodal attack vectors
- [ ] v1.0 — Full Glacis attestation integration, compliance reporting

## Acknowledgments

The evolutionary keep/discard loop is directly inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). autoredteam applies the same pattern to adversarial evaluation: where autoresearch evolves a research paper toward a quality metric, autoredteam evolves attack prompts toward a multi-dimensional vulnerability score.

## License

[Apache 2.0](LICENSE) — patent-friendly, important given Glacis's patent portfolio.

---

Built by [Glacis](https://glacis.io). The open-source tool is free forever. Cryptographic attestation is the paid upgrade.
