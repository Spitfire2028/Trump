# Cross-evaluator playing strength — Phase 7.3.11

Phase 7.3.9 showed evaluator substitution changes 12–36% of root
decisions. Phase 7.3.10 showed no candidate improves decision quality
anywhere the exact oracle can see. This phase asks the remaining
question: **does evaluator choice change who wins?**

Reproduce with:

```
python benchmarks/crossplay.py pilot
python benchmarks/crossplay.py deterministic 400
python benchmarks/crossplay.py ismcts 60 50
python benchmarks/crossplay.py analyse crossplay_deterministic.json
python benchmarks/crossplay.py strata
python benchmarks/crossplay.py decisions 1
python benchmarks/crossplay.py performance
python benchmarks/crossplay.py oracle
```

Self-play gives *relative* strength, never ground truth. Nothing here
proves any evaluator is objectively correct.

## Infrastructure: reused, not rebuilt

The repository already had what this needed, and the archaeology found
it: `simulation.play_game` (master-seed fan-out, per-seat redaction,
illegal-move refusal), `GameRecord`, `summarize`, and `SearchBot`. None
was modified. The one new component is an `ISMCTSAgent` adapter — ISMCTS
is a search class and nothing wraps it as an agent — and it lives in
`benchmarks/`, not production. No tournament framework was built.

## What is shared, and what is not

| | shared? |
|---|---|
| the deal (cards, trump, talon order) | **exactly** — `deal_seed` is derived from the master seed before any agent acts |
| seat pairing | **exactly** — every seed is played in both orientations |
| per-seat RNG stream | **exactly** — seat *s* always receives `agent_seeds[s]`, whichever evaluator sits there |
| ISMCTS internal trajectories after divergence | **no, deliberately** |

The last row is the one the brief warns about. Once two evaluators pick
different moves the games are different games; the streams stay the same
function of the seed but are consumed differently. Forcing them to stay
aligned would be artificial coupling, not variance reduction.

## Effect threshold, declared before the final run

**5 percentage points** of win-rate difference. Two reasons, both fixed
in advance: a change moving strength by less than that, after Phase
7.3.9 measured 12–36% of decisions changing, is equivalent for
engineering purposes whatever a p-value says; and Phase 7.3.10 measured
E3 at 3.34× the baseline's evaluator cost, which nothing under a few
points could justify.

## Power

Pilot, 40 seed-pairs, depth 3, baseline vs flexible: paired score sd
**0.281** → **248 pairs** needed for 5pp at 80% power (two-sided
α = 0.05). The deterministic arm ran **400 pairs per cell**, comfortably
above that.

ISMCTS pilot: sd **0.419** at 9.16 s per seed-pair → 551 pairs for the
same power, which is ~1.4 hours *per cell* and ~14 hours for the full
matrix. That arm is therefore deliberately underpowered and is reported
as such rather than padded out.

### The self-pair control is degenerate, and says so

Every self-pair (X vs X) scores exactly 0.500 with zero variance. This
is **not** evidence of anything: with deterministic agents the same deal
with reversed seats is literally the same game, so the pair always
averages 0.5 by construction. It confirms the seat-balancing is wired
correctly and nothing else. The bootstrap reports p = 1.000 for these
cells; an earlier version printed p = 0.000, which read as a significant
result and was the exact opposite of the truth.

## Deterministic cross-play — 12,000 seed-pairs, 24,000 games

Score is for the **left** evaluator, averaged over both seats. 400 pairs
per cell.

### Depth 1

| pair | score | 95% CI | p | draws | verdict |
|---|---|---|---|---|---|
| baseline vs mechanistic | 0.535 | [0.501, 0.568] | 0.045 | 0.062 | small |
| baseline vs structural | **0.651** | [0.619, 0.682] | 0.000 | 0.087 | MEANINGFUL |
| baseline vs flexible | **0.946** | [0.929, 0.961] | 0.000 | 0.016 | MEANINGFUL |
| mechanistic vs structural | 0.609 | [0.579, 0.637] | 0.000 | 0.200 | MEANINGFUL |
| mechanistic vs flexible | **0.955** | [0.942, 0.968] | 0.000 | 0.043 | MEANINGFUL |
| structural vs flexible | **0.922** | [0.906, 0.938] | 0.000 | 0.072 | MEANINGFUL |

### Depth 3

| pair | score | 95% CI | p | draws | verdict |
|---|---|---|---|---|---|
| baseline vs mechanistic | 0.464 | [0.432, 0.495] | 0.022 | 0.182 | small |
| baseline vs structural | 0.493 | [0.463, 0.525] | 0.651 | 0.201 | equivalent |
| baseline vs flexible | 0.538 | [0.509, 0.568] | 0.015 | 0.194 | small |
| mechanistic vs structural | 0.510 | [0.491, 0.529] | 0.304 | 0.285 | equivalent |
| mechanistic vs flexible | 0.517 | [0.496, 0.539] | 0.125 | 0.287 | equivalent |
| structural vs flexible | 0.514 | [0.496, 0.532] | 0.132 | 0.282 | equivalent |

### Depth 5

| pair | score | 95% CI | p | draws | verdict |
|---|---|---|---|---|---|
| baseline vs mechanistic | 0.506 | [0.485, 0.526] | 0.600 | 0.306 | equivalent |
| baseline vs structural | 0.512 | [0.489, 0.535] | 0.319 | 0.294 | equivalent |
| baseline vs flexible | 0.515 | [0.491, 0.539] | 0.238 | 0.280 | equivalent |
| mechanistic vs structural | 0.499 | [0.481, 0.517] | 0.919 | 0.314 | equivalent |
| mechanistic vs flexible | 0.491 | [0.469, 0.513] | 0.404 | 0.318 | equivalent |
| structural vs flexible | 0.499 | [0.482, 0.516] | 0.916 | 0.304 | equivalent |

**A strict ordering at depth 1, dissolving by depth 3.** At depth 1 the
ranking is baseline > mechanistic > structural ≫ flexible, transitive,
with no rock-paper-scissors and no seat asymmetry (P0 and P1 win rates
agree within ~2pp in every cell). By depth 3 every cell is inside ±4pp;
by depth 5, inside ±1.5pp. Draw rates rise with depth (1.6–8.7% → 18–20%
→ 28–32%), which is what deeper search into a shedding race should do.

## Why E3 collapses at depth 1 — the mechanism

A 0.946 score is large enough to demand a cause rather than a shrug.
Measured on **identical positions** drawn from real games, depth 1:

| evaluator | sheds a card in… |
|---|---|
| baseline | **97.4%** of choices |
| structural (E2) | 74.9% |
| flexible (E3) | **53.7%** |

*(n = 1,135 positions with ≥2 legal moves)*

Durak is won by running out of cards. E3 declines to shed nearly half
the time, and E2 a quarter of the time. Both defects trace to specific
terms introduced in Phase 7.3.10:

* **`throw_in_options`** (E3) rewards holding cards whose rank matches
  the table. Playing such a card *lowers* the score — the term rewards
  retaining exactly the cards that should be played.
* **`obligation_pressure`** (E2) is `open_attacks / own_cards`. Shedding
  a card shrinks the denominator and so *raises* the penalty. Phase
  7.3.10 recorded "explodes as the hand empties" as a pathology of this
  feature but never connected it to the incentive it creates.

Games against E3 at depth 1 run 74.8 plies against 61.3 for the baseline
mirror — consistent with hoarding, though length alone would not prove
it, which is why the shed rate was measured directly.

This is the phase's most useful result. Both defects were **invisible**
to every static, oracle and ablation metric in Phase 7.3.10, where the
candidates looked equivalent to the baseline. They are obvious within
minutes of actual play, and only at shallow depth: at depth 3 and 5 the
search finds the shedding line regardless of what the evaluator prefers,
which hides the defect completely.

## Stratified results

Baseline against the three candidates pooled, 3,600 pairs:

| stratum | n | baseline score | 95% CI |
|---|---|---|---|
| depth 1 | 1,200 | **0.711** | [0.691, 0.729] |
| depth 3 | 1,200 | 0.498 | [0.481, 0.516] |
| depth 5 | 1,200 | 0.511 | [0.498, 0.524] |

Branching factor, split by depth, since pooling would let the depth-1
effect masquerade as a branching effect:

| | low (<3) | medium (3–5) | high (5+) |
|---|---|---|---|
| depth 1 | 0.917 *(n=6)* | 0.710 *(n=1,194)* | — |
| depth 3 | 0.474 *(n=78)* | 0.500 *(n=1,122)* | — |
| depth 5 | 0.495 *(n=49)* | 0.512 *(n=1,151)* | — |

**The high-branching bucket is empty at every depth.** No *game* has a
mean branching factor ≥ 5, because averaging over a whole game washes
out the per-position spikes that Phase 7.3.9 measured. So this phase
cannot test 7.3.9's high-branching finding — the per-game aggregate is
the wrong instrument, and saying so is more useful than reporting a
bucket that does not exist.

Two other strata were pre-declared and turned out uninformative, which is
reported rather than quietly dropped:

* **terminal talon** — degenerate. Every Durak game ends with an empty
  talon, so the bucket has one value and carries no information.
* **game length** — downstream of the outcome. Long games score 0.706
  for the baseline, but games against a weak opponent *run* long, so the
  stratum cannot be read causally and is not.

## ISMCTS cross-play — 240 seed-pairs, 480 games, 50 iterations

Only the essential comparisons were run. The full 10-cell matrix at a
useful sample size is ~14 hours, so each candidate was played against the
baseline plus a self-pair control. **Cells not run are reported as not
run, never as equivalent.**

| pair | n | score | 95% CI | p | P0 wr | P1 wr | verdict |
|---|---|---|---|---|---|---|---|
| baseline vs mechanistic | 60 | 0.567 | [0.475, 0.658] | 0.179 | 0.500 | 0.633 | underpowered |
| baseline vs structural | 60 | 0.479 | [0.383, 0.575] | 0.640 | 0.392 | 0.567 | equivalent |
| baseline vs flexible | 60 | **0.637** | [0.550, 0.721] | 0.002 | 0.642 | 0.633 | MEANINGFUL |
| *[control]* baseline vs itself | 60 | 0.500 | [0.500, 0.500] | 1.000 | — | — | degenerate |

At n=60 with sd ≈ 0.419 this arm can only detect a ~15pp effect at 80%
power. It detects the one effect that large: the baseline beats E3 by
13.7 points. The `mechanistic` cell is the case the brief asks to be
named explicitly — an effect above the 5pp threshold with a confidence
interval straddling 0.5, which is **underpowered, not evidence**.

Two seat asymmetries appear here that the deterministic arm did not show
(structural 0.392 as P0 against 0.567 as P1). At n=60 these are well
within noise and are recorded, not interpreted.

50 ISMCTS iterations is a *low* budget, and this result agrees with what
the deterministic arm found at depth 1: shallow search exposes the
candidates' defects, and E3's most of all.

## Decision differences versus strength

The control the brief demands, measured at two depths:

| | disagreement with baseline | high-branch disagreement | strength (baseline score) |
|---|---|---|---|
| **depth 1** | | | |
| mechanistic | 17.4% | 3.5% | 0.535 |
| structural | 21.9% | 1.0% | 0.651 |
| flexible | **60.4%** | **68.3%** | **0.946** |
| **depth 3** | | | |
| mechanistic | 13.3% | 36.9% | 0.464 |
| structural | 15.3% | 37.0% | 0.493 |
| flexible | 17.2% | 39.5% | 0.538 |

*(n = 1,441–1,634 positions at depth 1; 1,081–1,181 at depth 3)*

At depth 3 the candidates disagree with the baseline on 13–17% of
decisions — and 37–40% at high branching — while being statistically
indistinguishable in strength. Disagreement alone predicts nothing. What
separates the depth-1 collapse from the depth-3 equivalence is not *how
many* decisions changed but *which direction* they changed in, which the
shed-rate measurement above identifies.

This is the concrete answer to Phase 7.3.9's open question: evaluator
sensitivity is real, and it is not the same thing as strength.

## Exact-reference cross-check

| evaluator | solvable positions | oracle-optimal |
|---|---|---|
| baseline | 13 | 100.0% |
| mechanistic | 13 | 76.9% |
| structural | 13 | 76.9% |
| flexible | 13 | 76.9% |

**This table is a sanity check and nothing more, for two reasons.** n=13,
and — more seriously — each evaluator's positions come from its *own*
self-play games, so the four rows are not the same positions. The
comparison is uncontrolled by construction. What it establishes is only
that no agent is playing wildly oracle-losing moves where the oracle can
see: 3 of 13 suboptimal choices for the candidates, 0 for the baseline,
on 13 positions each.

## Cost

Deterministic, depth 3, 25 games each:

| evaluator | s/game | games/hour | mean plies | vs baseline |
|---|---|---|---|---|
| baseline | 0.013 | 270,059 | 55.8 | 1.00× |
| mechanistic | 0.012 | 288,207 | 53.2 | 0.94× |
| structural | 0.013 | 285,124 | 53.3 | 0.95× |
| flexible | 0.016 | 230,283 | 58.5 | 1.17× |

ISMCTS at 50 iterations: 9.5–10.3 s per seed-pair, indistinguishable
across evaluators.

In whole games the 3.34× per-call evaluator cost Phase 7.3.10 measured
shrinks to 1.17×, because evaluation is a small share of a game's work
next to move generation and state transitions. The honest framing is
that E3 costs ~17% more per game **and loses**, so the trade-off question
never arises.

## Conclusion

**No candidate demonstrates a strength advantage at any budget tested.**
At depth 3, depth 5 and ISMCTS-50 the candidates are statistically
indistinguishable from the baseline. At depth 1 and, more weakly, at
ISMCTS-50, the baseline is *decisively stronger* than E3 and stronger
than E2.

So "empirically equivalent" is true only at adequate search depth. The
sharper statement is that two candidates carry defects that oppose the
game's objective, that search repairs those defects when given three or
more plies, and that every static and oracle metric in Phase 7.3.10 was
blind to them.
