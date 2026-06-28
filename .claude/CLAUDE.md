# Regime Trader — Claude Code Workflow

**Owner**: Ryan  
**Last Updated**: 2026-06-09  
**Model Context**: Refer to `.claude/memory/` for project state, past decisions, and outstanding tasks.

---

## Workflow: The 6 Power Phrases

This project uses a structured workflow to ensure specs are right before building, and verification happens at each gate. **Read these carefully** — they define how we work together.

### 1. **Launch sub agents**
Split independent tasks across parallel Claude sessions (one context per agent) instead of sequentially. Used for:
- Getting multiple perspectives on the same architectural problem
- Running truly independent tasks (e.g., backtest audit + scanner audit in parallel)
- Avoiding context pollution (earlier work biasing later thinking)

**When to use**: Multi-phase work, ambiguous problems, need fresh perspectives

### 2. **Write me an implementation spec**
Before coding, produce a detailed spec with decision points. **Key line to include**: *"For each step, show me the key decisions you'd make."*

A spec is not overhead — it's the first hour of building that saves 6 hours of rework.

**When required** (spec is non-negotiable):
- Strategy changes (HMM tuning, regime thresholds, allocation rules)
- Risk manager updates (circuit breakers, position sizing)
- Major refactors or new modules
- Anything longer than 2 hours of implementation

**When optional** (can skip):
- Bug fixes with clear root cause
- Refactors with tight scope
- Tactical changes <1 hour

**Spec format**:
- Goal (1-2 sentences)
- Key decisions (per step)
- Files affected
- Risks/blockers
- Backtest / validation strategy (if applicable)

### 3. **Interview me**
Instead of you writing a brief, I ask YOU the questions. This pulls out details you didn't know you needed to specify, then produces a spec.

**When to request**: When you have a problem but haven't fully thought through the solution.

### 4. **Verify before you build**
Three-layer approach:

**Layer 1 — Verification plan in CLAUDE.md** (this document)
- You define "human validation zones" — high-stakes decisions requiring sign-off
- You define success criteria upfront

**Layer 2 — Self-correction during build**
- I use tools to see my own output and validate
- I run tests before committing, backtests before going live

**Layer 3 — Human sign-off gates**
- See "Verification Gates" section below

### 5. **Based on this conversation, build me a skill**
A Claude skill is a reusable instruction set for repeating a process. Build only from validated conversations — never abstractly.

**When to build**: After you've validated a workflow and want it repeated reliably.

### 6. **Automate this** (the dangerous one)
Before automating, apply two filters:

**Taste test**: Does judging the output require human taste/judgment?
- YES → augment (build a tool that helps humans decide), not automate
- NO → safe to automate

**80/20 output analysis**: Would 80% quality be acceptable?
- YES → automate
- NO → augment or don't do it

**You must apply these filters** — I won't. I'll cheerfully automate bad decisions if you ask.

---

## Verification Gates (Human Sign-Off Required)

These decisions must be approved before code ships or goes live:

- [ ] **Strategy changes** (HMM tuning, regime thresholds, allocation profiles) — backtest validation + manual approval
- [ ] **Risk manager updates** (circuit breakers, drawdown limits, position size caps) — impact analysis before deployment
- [ ] **Live trading mode activation** (paper → live, live account mode changes) — explicit approval + account size verification
- [ ] **Wheel strategy deployment** (enabling for equities) — options chain validation + IV gate testing
- [ ] **Data feed / broker changes** (IEX → SIP, Alpaca SDK updates) — impact analysis + unit test pass
- [ ] **Alert/notification changes** (new alert types, messaging) — spam/alert fatigue review
- [ ] **Deployment to live account** — explicit manual approval

For all others: test pass + code review is sufficient.

---

## Spec Requirements by Project Phase

### Wheel Strategy
**Status**: Module complete (scanner built, pending integration)  
**Current phase**: Scanner → Live loop integration

**Specs needed**:
- Position entry logic (when scanner candidates → actual trades)
- Position management (exit rules, loss limits, profit targets)
- Integration with risk manager (sizing under circuit breakers)

**Verification gates**:
- Options data flow validation (Alpaca chain availability)
- IV gate testing (min IV rank enforcement)
- Collateral calculation accuracy

### Portfolio Review Logic
**Status**: Design phase (quant vs agent decision pending)

**Specs needed**:
- TBD: quant rules-based or agent-powered?
- Nightly vs intraday review frequency
- Integration with live loop and alerts

### Trade Review Agent (AI)
**Status**: Planning phase

**Specs needed**:
- Input schema (scanner result + market context)
- Output format (proceed / reduce_size / skip decisions)
- News/catalyst data sources

### Live Equity Trading
**Status**: Ready (infrastructure exists, strategy disabled in live mode)

**Specs needed**:
- When to enable (market condition gates)
- Entry signal overrides (regime + IV constraints)
- Position exit rules per regime

---

## Tech Stack & Key Context

**Core Stack**:
- Python 3.12
- Alpaca broker (paper + live trading)
- Hidden Markov Model (regime classification)
- pytest (test suite, 900+ tests)

**Modules**:
- HMM engine — 5-state regime classification
- BTC cycle engine — 60-day probabilistic cycle detection
- Risk manager — 4-layer circuit breaker system
- Scanner — nightly S&P 500 / Nasdaq 100 candidate selection
- Wheel scanner — new module for options income candidates
- Backtester + walk-forward engine — offline validation

**Data Sources**:
- Alpaca historical bars (OHLCV)
- Alpaca options chain (IV, delta, bid-ask, OI)
- yfinance (earnings calendar, market cap)

**Key Files**:
- `core/hmm_engine.py` — regime classification
- `core/cycle_engine.py` — BTC cycle detection
- `core/risk_manager.py` — circuit breaker logic
- `config/settings.py` — all tunable parameters
- `core/alerts.py` — notifications (email, webhook)
- `main.py` — live trading loop

---

## Standing Instructions

### Before Starting Any Task

1. **Check memory** — read `.claude/memory/MEMORY.md` for project state, past decisions, known issues
2. **Is this a spec phase or build phase?**
   - Spec: produce spec, show key decisions, request approval
   - Build: implement against approved spec, validate gates, update CLAUDE.md progress
3. **Does this require human sign-off?** (see Verification Gates)
   - If yes: flag it before building, wait for approval
   - If no: build, test, request review

### Code Standards

- **No comments unless WHY is non-obvious** — well-named code is self-documenting
- **No error handling for impossible scenarios** — trust framework guarantees
- **Prefer editing existing files** to creating new ones
- **No backwards-compatibility hacks** — if something is unused, delete it
- **Defensive at system boundaries only** — validate user input, external APIs; trust internal code

### Git Discipline

- **Never force-push to main** — confirm before any destructive git operation
- **Create new commits, don't amend** — unless user explicitly requests it
- **No skipping hooks** (`--no-verify`) or bypassing signing without explicit permission
- **Commit message should explain WHY**, not WHAT (code shows what; commit explains why)

### Testing

- **pytest passes before any merge** — run full suite (900+ tests expected)
- **Backtester validates before going live** — strategy changes need backtest proof
- **Type checking + tests verify correctness**, not feature correctness
- **Monitor for regressions** when modifying shared modules (HMM, risk manager, alerts)

---

## How to Correct Me

If I'm doing something wrong:
- **Tell me directly** ("don't do X, do Y instead")
- **Point to precedent** ("like you did for the cycle engine")
- **Explain the pattern** ("we gate risky decisions with verification gates")

I will save the correction to memory for future conversations.

---

## When to Escalate / Ask Questions

- **Ambiguous requirements** → request interview
- **Spec too vague** → produce spec anyway + flag assumptions
- **Unsure if verification gate applies** → ask before building
- **Multiple valid approaches** → present recommendation + tradeoff
- **Blocker or unknown unknown** → ask, don't guess

---

## Progress Tracking

**Current Active Work**:
- [ ] Wheel scanner: live loop integration (accept scanner output, filter by regime)
- [ ] Portfolio review logic: design phase (quant vs agent decision)
- [ ] Trade review agent: pending architecture design
- [ ] Duplicate alert dedup: COMPLETE (cycle_signal alerts fixed)

**Completed**:
- [x] Wheel scanner module (standalone, fully functional)
- [x] yfinance added to dependencies
- [x] Cycle signal alert deduplication (fingerprint-based)
- [x] HMM regime classification (5 states)
- [x] BTC cycle engine (60-day timing)
- [x] Risk manager (4-layer circuit breakers)
- [x] Full test suite (900+ tests)

---

## Notes for Next Session

- Wheel strategy integration: how to wire scanner output → position entries
- Portfolio review: decide on quant rules vs agent-powered approach
- Trade review agent: design spec needed before building
- Consider launching sub-agents for parallel work (trade agent design + portfolio logic)
