# Evaluator sensitivity & causal impact — Phase 7.3.9

A controlled study of one question: **when the leaf evaluator is swapped
and everything else is held identical, how often does the search actually
decide differently?**

Reproduce with:

```
python benchmarks/eval_sensitivity_corpus.py
python benchmarks/eval_sensitivity.py deterministic
python benchmarks/eval_sensitivity.py strata
python benchmarks/eval_sensitivity.py ismcts
python benchmarks/eval_sensitivity.py reference
python benchmarks/eval_sensitivity.py leaves        # completion pass
python benchmarks/eval_sensitivity.py amplitude     # completion pass
python benchmarks/eval_sensitivity.py margins       # completion pass
python benchmarks/eval_sensitivity.py inert         # completion pass
python benchmarks/eval_sensitivity.py performance   # completion pass
```

The five stages marked *completion pass* were added after the first
write-up, together with the V3/V4 amplitude probes. They required **no
production change**: the leaf counters they read already existed in
`SearchStats` and were simply never looked at.

> Decision sensitivity is **not** playing strength. Nothing here says any
> variant plays better. High sensitivity does not mean the evaluator is
> wrong, and low sensitivity does not mean it is right.

## The injection point

`world_search` already resolves `get_evaluator(config.evaluator)`, so the
deterministic experiment needed **no production change**: variants are
registered in the existing `EVALUATORS` registry and selected by name.

ISMCTS was different. `ISMCTSConfig.evaluator` was declared and
documented but **never used** — the leaf call hardcoded the baseline. The
field had no effect. Phase 7.3.9 wired it through (resolved once, eagerly,
so a misspelled name fails at construction). The default resolves to the
same `BaselineEvaluator` as before, so existing callers are unaffected;
a regression test asserts that.

## Evaluator variants

| | definition |
|---|---|
| **E0** | the production `BaselineEvaluator`, untouched |
| **E1** | card advantage only (`own_trump = 0`, `open_obligation = 0`) |
| **E2** | no card advantage (`card_advantage = 0`), other terms kept |
| **E3** | baseline terms, card advantage halved while the talon is non-empty |
| **V3** | baseline terms, `own_trump` amplified 0.30 → 3.0 (×10) |
| **V4** | baseline terms, `open_obligation` amplified 0.50 → 3.0 (×6) |

V3 and V4 are deliberately implausible. Tripling a coefficient is not a
proposal; it asks whether the search can be moved by that term *at all*,
at an amplitude far beyond anything defensible. Both keep the baseline's
sign, and both stay far inside `MATE_MARGIN` (bounds 66.0 and 56.7
against a margin of 1000), so neither can be mistaken for a proven
result. A high change rate from V3 means the trump term can reach the
decision — **not** that more trump weight is better.

E3's discount is a single a-priori structural choice, motivated by
Phase 7.3.8's observation that card advantage is near-meaningless while
both hands refill to six. **No coefficient anywhere was tuned to a
result.**

### Information classification

Every variant scores a `SearchNode`, which represents the opponent as a
**count** and the talon as a **size**. An evaluator built on it is
structurally incapable of reading a hidden hand or hidden talon contents.

| feature | class |
|---|---|
| own hand, own trumps | OWN_PRIVATE |
| opponent card count, talon size, table, trump suit, phase, attack limit | PUBLIC |
| card advantage, obligation, talon discount | DERIVED |
| — | no HIDDEN_WORLD_ONLY feature exists |

## Classification rule, fixed before any results

With `m0` the move E0 picks and `mk` the move Ek picks:

```
Type T (tie-break artifact)  |V0(m0) - V0(mk)| <= TOL
                        and  |Vk(m0) - Vk(mk)| <= TOL
Type R (genuine ranking)     otherwise
```

`TOL = 1e-9`. Terms are coefficients of order 0.3–1.0 and terminal scores
of order 10⁴, so anything closer is float noise.

## Deterministic search (220 positions, true-world determinization)

Genuine ranking changes versus E0:

| depth | E1 card-only | E2 no card advantage | E3 talon-aware | V3 strong trump | V4 strong obligation |
|---|---|---|---|---|---|
| 1 | 20.9% | 25.9% | 1.4% | 21.4% | 0.0% |
| 3 | 13.2% | 29.5% | 1.8% | 16.4% | 0.0% |
| 5 | 12.3% | 35.9% | 3.6% | 15.5% | 0.0% |

E1/E2/E3 reproduce the original run exactly, which is the reproducibility
check the harness promises.

**Type T count: 0 at every depth.** Every observed change was a genuine
ranking change.

That deserves an explanation, because 50% of root positions *do* have an
exact top-value tie (median best-vs-second margin 0.150). Ties are
resolved by a deterministic, evaluator-independent order, so both
variants break a tie the same way and the decision never changes because
of it. Ties therefore cannot inflate these rates.

Sensitivity to the *dominant* feature (E2) **grows** with depth, and its
mean value gap grows with it (1.04 → 1.37 → 3.12). Sensitivity to the
*auxiliary* features (E1) **shrinks** with depth.

## Stratified (depth 3, pooled variants)

| stratum | n | ranking change |
|---|---|---|
| 2–3 legal moves | 366 | 6.8% |
| 4–6 legal moves | 258 | 23.6% |
| 7+ legal moves | 36 | 33.3% |
| early / mid / late | 126 / 396 / 138 | 19.0% / 14.4% / 12.3% |
| talon 7+ / 1–6 / empty | 363 / 159 / 138 | 16.8% / 12.6% / 12.3% |
| attacker / defender | 417 / 243 | 14.6% / 15.2% |

Branching factor dominates. Phase, talon size and seat barely move the
rate, and the 7+ bucket (n=36) is small enough to treat as indicative
only.

## ISMCTS (40 positions, common random numbers)

Same seed, same budget, same determinization stream; only the evaluator
differs. TVD is total variation distance between normalised root visit
distributions, defined before the run.

| iterations | | E1 | E2 | E3 | V3 | V4 |
|---|---|---|---|---|---|---|
| 50 | changed | 10.0% | 65.0% | 20.0% | 32.5% | 0.0% |
| | TVD mean | 0.077 | 0.387 | 0.161 | 0.174 | 0.000 |
| 200 | changed | 17.5% | 70.0% | 20.0% | 30.0% | 0.0% |
| | TVD mean | 0.093 | 0.476 | 0.137 | 0.272 | 0.000 |
| 800 | changed | 12.5% | 75.0% | 22.5% | — | — |
| | TVD mean | 0.086 | 0.530 | 0.146 | — | — |

E1/E2/E3 reproduce the original run exactly. V3/V4 were measured at 50
and 200 iterations in the completion pass; the 800-iteration cell was not
re-run for them.

ISMCTS is **more** evaluator-sensitive than deterministic search, and E2's
sensitivity **rises** with budget rather than washing out. E3, nearly
inert under deterministic search (1.8%), reaches 20% here — the evaluator
shapes which subtrees get visited, not merely a final comparison.

## Terminal versus depth-cutoff leaves (160 positions)

The measurement that says whether the evaluator can matter at a given
depth at all. A leaf the evaluator scored is a *cutoff* leaf; a leaf the
frozen engine proved is a *terminal*.

| depth | leaves | terminal | cutoff | cutoff share | nodes |
|---|---|---|---|---|---|
| 1 | 559 | 0 | 559 | 100.0% | 1,118 |
| 2 | 1,501 | 0 | 1,501 | 100.0% | 2,619 |
| 3 | 3,224 | 1 | 3,223 | 100.0% | 5,843 |
| 4 | 6,900 | 6 | 6,894 | 99.9% | 12,973 |
| 5 | 13,026 | 13 | 13,013 | 99.9% | 26,738 |

**Essentially every leaf at every tested depth is scored by the
evaluator.** 13 terminals out of 13,026 leaves at depth 5. Whatever else
is true, low sensitivity here could not be blamed on the search resolving
positions tactically before evaluation matters — at these depths it
almost never does. This is the structural precondition for evaluator
quality being load-bearing, and it is met.

## Perturbation amplitude at shared leaves

For every leaf a baseline depth-3 search actually scored (2,570
non-terminal leaves from 120 positions), `delta = E_variant(leaf) −
E_baseline(leaf)`. Terminals are excluded: every variant scores those
identically by construction, so including them would pad the
distribution with structural zeros.

| variant | mean \|Δ\| | median \|Δ\| | p95 \|Δ\| | max \|Δ\| | mean \|Δ\| ÷ median margin |
|---|---|---|---|---|---|
| E1 card-only | 0.259 | 0.300 | 0.600 | 1.200 | 0.86 |
| E2 no card advantage | 3.284 | 3.000 | 7.000 | 11.000 | 10.95 |
| E3 talon-aware | 1.357 | 1.000 | 3.500 | 5.500 | 4.52 |
| V3 strong trump | 2.561 | 2.700 | 8.100 | 10.800 | 8.54 |
| V4 strong obligation | 0.350 | **0.000** | 2.500 | 2.500 | 1.17 |

Median root best-vs-second margin at depth 3: **0.300** (n=120).

The last column is the causal link the phase was built to establish:
leaf perturbation → root margin → decision. E2 and V3 move a typical leaf
by 8–11× the typical root margin, and both change 15–36% of decisions.
E1 moves a leaf by less than one margin and changes 12–21%. V4's *median*
delta is zero, because the obligation term is guarded and does not fire
on most leaves.

## Root-margin analysis (160 positions, depth 3, variants pooled)

Margin is always measured under the **baseline**, so which bucket a
position falls into does not depend on the variant being tested.

Margin under the baseline: median 0.300, mean 63.439, max 9,996.000. The
mean is meaningless — it is dragged by forced wins scoring near ±10,000 —
and the median is the statistic to read.

| margin bucket | n (position × variant) | ranking change |
|---|---|---|
| tie (≤1e-9) | 355 | 8.2% |
| 0 – 0.5 | 90 | 23.3% |
| 0.5 – 1.5 | 115 | 14.8% |
| 1.5+ | 240 | 5.8% |

Ranking changes concentrate where the baseline's decision was close, and
fall by a factor of four as the margin grows. But they do **not** vanish:
5.8% of clearly separated decisions (margin ≥ 1.5) were still overturned.
So the answer to the either/or in the phase brief is *both*, with the
weight on close decisions.

The tie bucket needs a word, since a ranking change inside it looks
contradictory. A tied margin means the baseline's top *two* moves are
equal; a variant can still select a third move that the baseline strictly
disprefers, which is a genuine ranking change by the fixed rule.

## Why V4 changes leaf values yet never changes a decision

V4 changing 0 of 220 decisions at every depth is either a dead probe or a
real finding, and the difference matters. Measured, in order:

| measurement | V3 strong trump | V4 strong obligation |
|---|---|---|
| leaves where the variant differs from the baseline | 1,615 (62.8%) | 360 (14.0%) |
| leaves where the obligation term is even applicable | — | 360 (14.0%) |
| positions where a root *value* moved | 90 of 120 | 56 of 120 |
| …of those, the *pick* was unchanged | 75 | **56 (all)** |
| mean within-position spread of the per-move delta | 2.419 | 2.379 |
| delta was a pure common offset | 12 of 90 | **0 of 56** |
| baseline's own pick received the largest delta | 50 of 90 (55.6%) | **56 of 56 (100.0%)** |

The probe is not dead: V4 fires on exactly the 14.0% of leaves where the
baseline's own guard admits the term, and it moves root values in 56 of
120 positions by a within-position spread averaging 2.38 — eight times
the median margin.

My first explanation was that the penalty acts as a common offset across
sibling moves, which cannot reorder an argmax. **That is wrong**, and the
measurement says so: 0 of 56 deltas were a pure offset.

What actually happens is that the perturbation is perfectly aligned with
the ranking it perturbs. In all 56 positions where V4 moved anything, the
move the baseline had already chosen received the largest delta of any
root move. Amplifying the obligation term therefore reinforces the
baseline's existing preference rather than competing with it — the
baseline already contains that term, with the same sign, and in these
positions the move that minimises open obligations is already the move
the rest of the evaluation likes. V3 amplifies an existing term too, but
favours the baseline's pick only 55.6% of the time — near a coin flip —
and correspondingly changes 15–21% of decisions.

So amplification is not what produces sensitivity. *Disagreement* is.

This is 56 positions at depth 3. "100%" is what was observed, not a
theorem; it may well be a property of this corpus rather than of the
obligation term in general.

## Diagnostic performance (40 positions, wall clock, one process)

| depth | baseline seconds | positions/s | nodes | slowest variant |
|---|---|---|---|---|
| 1 | 0.01 | 6,299 | 296 | 1.05× |
| 3 | 0.04 | 950 | 1,507 | 1.04× |
| 5 | 0.22 | 181 | 7,418 | 1.04× |

All variants run within ±7% of the baseline; the spread is measurement
noise, not evaluator cost. The amplitude recorder (`rec_baseline`) visits
an **identical** node count at every depth — 296 / 1,507 / 7,418 — which
is the check that it observes the search without changing which leaves
are reached. No production code was optimised for any of this.

## Exact-reference comparison

The independent Phase 7.3.8 alpha-beta solver (frozen Phase 2 transitions
only — no `BaselineEvaluator`, no `world_search`, no ISMCTS) could solve
**116 of 900** corpus positions, and only the smallest.

| variant | oracle-optimal | mean move-value loss |
|---|---|---|
| E0 | 90.5% | 0.164 |
| E1 | 87.9% | 0.216 |
| E2 | 87.9% | 0.216 |
| E3 | 90.5% | 0.164 |

All four are nearly indistinguishable there. That is the Phase 7.3.8
limitation seen from the other side: **the oracle-reachable domain is
exactly the domain where the evaluator is not load-bearing**, because
depth-3 search already reaches terminals. This coverage cannot rank the
variants and is not used to.

## Findings

1. **Evaluator substitution causes genuine ranking changes at realistic
   rates** — 12–36% deterministic, up to 75% in ISMCTS — none of it
   attributable to tie-breaking.
2. **Sensitivity is concentrated by branching factor**, not by game
   phase or talon size.
3. **Depth does not wash out evaluator influence.** For the dominant
   feature it amplifies it. This qualifies the Phase 7.3.6 reading that
   search horizon dominates evaluator quality: horizon changed 29.1% of
   decisions, and removing the dominant evaluator feature changes a
   comparable or larger share.
4. **ISMCTS is more sensitive than deterministic search**, so any future
   evaluator work matters more there.
5. **Essentially every leaf at depths 1–5 is scored by the evaluator**
   (13 terminals in 13,026 leaves at depth 5), so the precondition for
   evaluator quality mattering is structurally met.
6. **Amplitude is not what produces sensitivity; disagreement is.** V4
   amplifies a term ×6 and changes nothing, because in all 56 positions
   where it moved a value it favoured the move the baseline had already
   chosen. V3 amplifies ×10 and changes 15–21%, favouring the baseline's
   pick only 55.6% of the time.
7. **Sensitivity concentrates in close decisions but is not confined to
   them**: 23.3% ranking change at margins under 0.5, still 5.8% at
   margins of 1.5 and above.

## Known limitations

- Corpus is 900 positions from 120 seeded trajectories; deterministic
  stage uses 220, ISMCTS 40, reference 116. Not all strata have adequate
  samples (7+ legal moves: n=36).
- The deterministic stage uses the *true* hidden world so that the
  evaluator is the only variable. That removes world noise; it is not a
  claim about play under uncertainty.
- E1/E2/E3 are diagnostic probes, not candidate production evaluators.
- No playing-strength measurement was made, and none is implied.

## Second pre-existing finding: the obligation term is dead inside ISMCTS (NON-BLOCKING)

V4's TVD of exactly 0.000 is too clean to report without checking that
the probe reaches ISMCTS at all. It does — and what the check found is
about production code, not about the probe.

Instrumenting the evaluator ISMCTS actually calls, over 40 positions at
200 iterations (7,591 leaf evaluations):

| at an ISMCTS leaf | share |
|---|---|
| table non-empty | 87.6% |
| searcher is the defender | 46.6% |
| phase is DEFENSE | 49.5% |
| at least one undefended slot | 56.1% |
| **`_facing_attacks` true** | **0 of 7,591 (0.0%)** |

Each condition is common; the conjunction never happens. The breakdown
shows why:

| phase | searcher's seat | to move | leaves |
|---|---|---|---|
| DEFENSE | attacker | OPPONENT | 3,760 |
| ATTACK | defender | OPPONENT | 3,166 |
| TAKING | defender | OPPONENT | 369 |
| ATTACK | attacker | SELF | 170 |
| TAKING | attacker | SELF | 126 |

ISMCTS evaluates on the **opponent's** turn in 7,295 of 7,591 leaves. So
whenever the phase is DEFENSE the defender is the opponent, which makes
the searcher the attacker; and whenever the searcher is the defender the
opponent is attacking, so the phase is ATTACK or TAKING. The guard
`not searcher_is_attacker and phase is Phase.DEFENSE` therefore cannot
be satisfied at any node ISMCTS scores.

**The production `BaselineEvaluator`'s third term contributes exactly
nothing in the ISMCTS path.** It is live in `world_search`, where it
applies at 14.0% of leaves, and dead in ISMCTS. That is why V4's visit
distributions are bit-identical to the baseline's: inside ISMCTS, V4
*is* the baseline.

This is a pre-existing property of code this phase freezes — the term,
the guard and the ISMCTS evaluation point all predate it — so it is
recorded, not fixed. It is also the sharpest reason not to read V4's 0.0%
as "the search is insensitive to obligations".

## Pre-existing finding, not introduced here (NON-BLOCKING)

ISMCTS evaluates the expanding iteration's leaf **before** incrementing
depth, so that one iteration scores a terminal at ply 0 while later
iterations score the same terminal at ply 1. Observed directly: a forced
win over 5 iterations returned 9999.2 = (4 × 9999 + 1 × 10000) / 5.

The effect is a one-ply discount inconsistency on the first visit to a
node, not a sign or perspective error. It originates in Phase 7.3.5,
which this phase's scope freezes, so it is recorded rather than fixed.
`test_terminal_values_carry_the_ply_discount` now pins the discount so
the behaviour cannot silently disappear.
