# Evaluation quality: Phases 7.3.7 and 7.3.8

Two measurement phases that ended without shipping an evaluator. Both
conclusions were negative, and both are recorded here because a negative
result that prevents building the wrong thing is worth keeping.

Reproduce with `python benchmarks/eval_study.py {accuracy,moves,ablation,tune}`
and `python benchmarks/talon_reference.py`.

## What BaselineEvaluator is

Three terms, from `ai/evaluation.py`:

| Feature | Source | Class | Weight |
|---|---|---|---|
| `card_advantage` | `opponent_cards - own_cards` | PUBLIC (counts) | 1.0 |
| `own_trump` | trumps in own hand | OWN_PRIVATE | 0.30 |
| `open_obligation` | undefended slots, when defending | PUBLIC | -0.50 |

Terminals short-circuit via `node.outcome()` and are never approximated.
There is **no talon term**, and the node exposes only an opponent card
*count*, so the evaluator structurally cannot see opponent trumps.

## Phase 7.3.7: accuracy against an exact endgame oracle

Oracle: `perfect_information_value` (exhaustive minimax over frozen
Phase 2 transitions; no evaluator, no production search). Domain chosen
by measured cost — empty talon, at most 6 cards. Corpus 321 positions,
split dev 160 / validation 80 / held-out test 81.

| split | correlation | sign accuracy | base rate |
|---|---|---|---|
| dev | +0.444 | 64.4% | 61.1% |
| validation | +0.394 | 58.8% | 61.1% |

**The evaluator is at or near the majority-class base rate.**

### Two of three features are inert

| variant | correlation | sign |
|---|---|---|
| baseline | +0.444 | 64.4% |
| no trump term | +0.442 | 64.4% |
| **trump term negated** | **+0.439** | **64.4%** |
| card advantage only | +0.414 | 64.4% |

Reversing a feature's sign changes nothing: 0.30 cannot outweigh 1.0 per
card. The "no card advantage" row (+0.496 correlation, 42.5% sign, 61.9%
flat scores) shows why a single metric misleads — correlation rises while
the distribution collapses to zeros.

### Retuning does not generalise

| split | baseline | tuned (0.25 / 3.0 / 1.5) |
|---|---|---|
| dev | 64.4% | 66.9% |
| validation | 58.8% | **68.8%** |
| **held-out test** | **72.8%** | **70.4%** |

Dev and validation improved; held-out got worse. **Rejected.** A
dev/validation improvement that fails held-out testing is a rejection,
not a success.

### Horizon

In the oracle domain, deeper search is monotonically better — best-move
agreement 84.6% (depth 0) to 92.3% (depth 2) to **100% (depth 5)** —
because depth 5 reaches actual terminals in <=6-card endgames. That
reframes the 7.3.6 finding: depth *corrects* the evaluator rather than
oscillating. It also exposes the limitation: the only exactly-solvable
domain is the one where the evaluator matters least.

### Antisymmetry

`|E(pos,P0) + E(pos,P1)|` mean 0.216, max 0.600; exactly antisymmetric in
only 11 of 31 positions. The violation equals `own_trump` times the
trump-count difference, because the term has no opponent counterpart.
Antisymmetry is **desirable but not contractually required**: search uses
a single fixed root-player perspective, so it is internally consistent.

## Phase 7.3.8: the talon-bearing domain does not exist

Two facts established first: `transposition_key` includes the talon and
excludes history, so memoised exact solving is sound; and a fixed talon
order makes the game **fully deterministic** (no chance node), since
draws take from a fixed end.

`benchmarks/talon_reference.py` is an independent alpha-beta solver over
frozen transitions with a node budget and an exactness flag. Verified
50/50 against the test-suite oracle on empty-talon endgames.

**Pruning mattered.** The first solver was full-width; adding alpha-beta
moved 12-card positions from 0/10 proven to 8/8 at 2.5 s mean. Declaring
infeasibility from an unpruned solver would have been an artefact.

| cards in play | proven | mean time |
|---|---|---|
| 10 | 4/5 | 5.1 s |
| 12 | 8/8 | 2.5 s |
| 13 | 5/8 | 8.1 s |

**The blocking fact:** when a talon is present, hands refill to 6, so
real talon-bearing positions have 13-36 cards in play. Only **0.92%**
have <=12. The solvable slice is ~1% of the domain by frequency and sits
at talon sizes 1-5 — the near-exhausted tail, not the mid-game where the
evaluator is load-bearing.

Corpus reached 31 proven positions in 171 s (held-out n=8). Action
ranking, horizon and ablation were **deliberately not reported** on it:
at that sample size they would be false precision.

**Verdict: D — domain still insufficient.** No credible talon-bearing
reference exists, so no evaluator redesign is justified. A deep-search
reference is circular, since it would depend on the evaluator under test.

## Next step recommended by 7.3.8

Do not pursue evaluator work through exact references. Measure *relative*
playing strength between evaluator variants in self-play — no ground
truth needed, only a comparison — which requires the tournament
infrastructure scheduled for Phases 12-13.
