# Principled evaluator design — full-brief pass (Phase 7.3.10)

Feature archaeology across every category the brief names, a candidate
*family* rather than one candidate, antisymmetry, the coefficient
question, and the held-out decision.

No playing-strength measurement was made and none is implied. Nothing
here says any evaluator plays better than any other.

Reproduce with:

```
python benchmarks/feature_study.py formalize
python benchmarks/feature_study.py distribution   --split dev
python benchmarks/feature_study.py redundancy     --split dev
python benchmarks/feature_study.py oracle         --split dev
python benchmarks/feature_study.py antisymmetry   --split dev
python benchmarks/evaluator_validation.py ranking       --split dev
python benchmarks/evaluator_validation.py family        --split dev
python benchmarks/evaluator_validation.py sweep         --split dev
python benchmarks/evaluator_validation.py latency       --split dev
python benchmarks/evaluator_validation.py ismcts_family --split dev
python benchmarks/evaluator_validation.py ply_bias      --split dev
python benchmarks/mutate_phase7310.py
```

---

## 1. The evaluation target, formally

`E(position, root_player)` is a **heuristic estimate, in the searching
player's units, of how favourable an undecided position is for
`root_player`**, on a scale that is bounded strictly inside
`MATE_MARGIN = 1000` so that no estimate can ever be confused with a
proven result.

It must fit four consumers, and it does:

* **Minimax.** One tree has one fixed `root_player`. The layer maximises
  at that player's nodes and minimises at the opponent's, so the
  evaluator is *not* side-to-move-relative and never needs negating.
* **Terminal values.** A decided node never reaches the heuristic;
  `terminal_score` handles it exactly, with a ply discount.
* **Multi-world aggregation.** Worlds are aggregated by value in a common
  root-player frame, which the above already guarantees.
* **ISMCTS backpropagation.** Same frame, same sign convention.

### Is antisymmetry required?

`E(pos, P0) = −E(pos, P1)` is **not** implied by that contract. A single
search tree carries one `root_player`; the two seats' evaluations are
never added, compared or combined anywhere in the engine. Antisymmetry is
therefore a *semantic* property — a claim that the evaluator measures one
zero-sum quantity rather than two unrelated ones — and not a correctness
requirement.

Measured over 500 fully known development positions
(`feature_study.py antisymmetry`), `|E(pos,P0) + E(pos,P1)|`:

| quantity | mean | median | p95 | max | exact zero |
|---|---|---|---|---|---|
| material_raw | 0.000 | 0.000 | 0.000 | 0.000 | **100.0%** |
| material_persistent | 0.000 | 0.000 | 0.000 | 0.000 | **100.0%** |
| seat | 0.000 | 0.000 | 0.000 | 0.000 | **100.0%** |
| terminal_proximity | 0.000 | 0.000 | 0.000 | 0.000 | **100.0%** |
| trump_count | 1.616 | 1.500 | 4.000 | 5.000 | 22.8% |
| trump_quality | 0.793 | 0.750 | 2.125 | 3.375 | 26.2% |
| open_obligation | 1.284 | 2.000 | 2.000 | 8.000 | 41.2% |
| defense_flexibility | 1.512 | 0.000 | 6.000 | 8.000 | 52.0% |
| talon_size | 16.592 | 12.000 | 46.000 | 48.000 | 28.6% |
| own_hand_size | 12.526 | 12.500 | 17.000 | 21.000 | 0.0% |
| **production baseline** | **0.403** | 0.300 | 1.200 | 1.500 | **18.0%** |
| **mechanistic candidate** | **0.403** | 0.300 | 1.200 | 1.500 | 18.0% |

The pattern is exact and explains itself. *Difference* features —
`opponent − own` in some form — are antisymmetric by construction, at
100.0% and zero error. *Own-side* features (trumps, flexibility) cannot
be, because each player counts only their own. *Shared* features are
worst of all: `talon_size` yields `|t + t| = 2t`, which is why its mean
error is 16.59 ≈ 2 × the mean talon size of 8.30.

So **the production evaluator is antisymmetric in only 18.0% of
positions**, with a bounded error (max 1.5) contributed entirely by the
own-trump and obligation terms. The consequence is semantic rather than
mechanical: a position can look mildly good for *both* players at once.
That is a real defect in interpretation, not a contract violation, and it
is inherited by every candidate here rather than silently "fixed" —
forcing antisymmetry would mean deleting the trump term, which is a
different proposal requiring its own evidence.

## 2. Feature archaeology

Nineteen features, one per category the brief names, each formalised with
definition, inputs, information class, range, normalisation, expected
sign, symmetry behaviour, cost and known pathologies
(`feature_study.py formalize`). Information classes present:
`PUBLIC` 5, `OWN_PRIVATE` 2, `OWN_PRIVATE + PUBLIC` 7,
`PUBLIC + OWN_PRIVATE` 5, **`HIDDEN_WORLD_ONLY` 0 — unconstructible**,
because `SearchNode` has no field that could hold a hidden card.

### One trap worth naming

`SearchNode.self_moves()` returns `()` whenever it is not the searcher's
turn. Phase 7.3.9 measured ISMCTS as evaluating on the *opponent's* turn
at 7,295 of 7,591 leaves, so a flexibility feature built on it would read
zero at essentially every ISMCTS leaf while looking perfectly healthy on
a corpus of root positions — where it is never zero (min 1). Every
flexibility feature here is therefore computed from the hand and the
public table, independent of whose turn it is, and
`turn_gated_moves` is kept as a control so the trap is visible.

### Distributions (development, n=500)

| feature | min | max | mean | stdev | zeros |
|---|---|---|---|---|---|
| material_raw | −15.00 | 12.00 | 0.030 | 4.087 | 13.0% |
| material_persistent | −15.00 | 12.00 | 0.052 | 3.748 | 24.2% |
| trump_count | 0 | 4 | 0.870 | 0.928 | 43.0% |
| trump_quality | 0 | 3.12 | 0.413 | 0.542 | 48.8% |
| distinct_ranks | 0 | 8 | 4.484 | 1.528 | 1.2% |
| throw_in_options | 0 | 4 | 0.542 | 0.777 | 61.2% |
| defense_flexibility | 0 | 7 | 0.840 | 1.309 | 60.6% |
| open_obligation | 0 | 4 | 0.642 | 0.595 | 41.2% |
| obligation_pressure | 0 | 2.00 | 0.127 | 0.189 | 41.8% |
| talon_size | 0 | 24 | 8.296 | 7.911 | 28.6% |
| refill_deficit | 0 | 4 | 0.276 | 0.590 | **78.4%** |
| terminal_proximity | −0.92 | 0.94 | −0.007 | 0.150 | 72.4% |

`refill_deficit` is zero in 78.4% of positions — mostly inert, and
dropped from every candidate on that evidence alone.

## 3. Redundancy

Pairwise |Pearson r| over the 500 development positions:

| pair | r | verdict |
|---|---|---|
| talon_size / talon_exhaustion | **−1.000** | redundant by construction (affine) |
| material_raw / material_persistent | **0.981** | see below |
| trump_count / trump_control | 0.869 | overlapping |
| trump_count / trump_quality | 0.848 | overlapping, not redundant |
| own_hand_size / distinct_ranks | 0.833 | overlapping |
| material_raw / own_hand_size | −0.826 | overlapping |
| open_obligation / obligation_pressure | 0.678 | related |
| **turn_gated_moves / defense_flexibility** | **0.098** | nearly independent |
| talon_size / refill_deficit | 0.183 | nearly independent |

Two results decide design choices.

**Correlation is the wrong lens for decision quality.** `material_raw`
and `material_persistent` correlate at r = 0.981, which by the usual
reading would make one of them redundant. They are not: they differ on
190 of 357 talon-bearing positions and change 14.3% of depth-3 root
decisions. A 0.98 correlation between two quantities says almost nothing
about whether an argmax over them agrees. The brief's own hierarchy
(§29) warns about exactly this, and here is the measurement behind it.

**Counting legal moves is not a proxy for defensive capability.**
`turn_gated_moves` and `defense_flexibility` correlate at r = 0.098.
They measure different things, and the brief's caution in §6D is
well-founded.

`talon_exhaustion` was dropped for genuine redundancy (r = −1.000).

## 4. Single-feature behaviour — and the control that reinterprets it

Each feature used alone as a one-term evaluator at the root, compared
against the independent Phase 7.3.8 exact solver on the 66 solvable
development positions:

| feature | oracle agreement |
|---|---|
| **throw_in_options** | **89.4%** |
| material_raw, material_persistent, trump_count, trump_quality, trump_control, seat, talon_size, refill_deficit, talon_exhaustion, turn_gated_moves | 83.3% |
| **control_constant (tie-break floor)** | **83.3%** |
| distinct_ranks, attack_room | 80.3% |
| defense_flexibility | 78.8% |
| defense_breadth | 75.8% |
| open_obligation, obligation_pressure | 74.2% |
| own_hand_size, terminal_proximity | 68.2% |

The ten-way tie at 83.3% looked like a result until the control was
added. `control_constant` returns 0 for every move, so its argmax always
falls to the first legal move under the harness's deterministic
tie-break — and it scores 83.3% too. Those ten features are **constant
across the sibling moves** of these positions (`talon_size` and `seat`
are shared or fixed; the material terms do not vary between siblings in
this domain), so their score measures the tie-break, not the feature.

With the floor established: exactly one feature beats it
(`throw_in_options`, 89.4%), and six score *below* it — in this domain
they actively mislead. Every number here comes from a domain Phase 7.3.8
showed to be almost entirely empty-talon, so this is a sanity check on
sign and liveness, never a feature ranking.

## 5. The candidate family

| | definition |
|---|---|
| **E0** `baseline` | production, untouched |
| **E1** `mechanistic` | persistent material (pass 1); all baseline coefficients inherited |
| **E2** `structural` | E1 + trump **quality** for trump count, obligation **ungated and normalised**, terminal proximity |
| **E3** `flexible` | E2 + throw-in options (attacking) and defensive answers (defending) |

E2's three changes each answer a measurement rather than an intuition:
trump quality because count and quality correlate at only 0.848 and the
baseline scores the six and the ace of trumps identically; the ungated
obligation because Phase 7.3.9 measured the baseline's guard as satisfied
at **0 of 7,591** ISMCTS leaves, making the production term dead in that
entire search path; terminal proximity because material alone does not
distinguish a two-card race from an eight-card one once refill has
stopped.

## 6. Coefficients — what was and was not fitted

**No coefficient was fitted to any outcome, and the reason is structural
rather than fastidious.** Fitting requires a target, and every available
target is disqualified:

* the exact solver reaches only ~13% of positions, almost all
  empty-talon — the region where E1 is provably *identical* to E0, so it
  cannot discriminate what would be fitted;
* a deep search using `BaselineEvaluator` as the target is circular, as
  Phase 7.3.8 established;
* game outcomes would require self-play, which this phase forbids.

What *was* done is scale-setting, which consults the distribution of the
features and never an outcome. One declared constant,
`AUXILIARY_SHARE = 0.10`, generates every auxiliary coefficient:

```
weight_i = AUXILIARY_SHARE * stdev(material) / stdev(feature_i)
```

giving trump_quality 0.69, obligation_pressure 1.98, terminal_proximity
2.50, defense_flexibility 0.29, throw_in_options 0.48 from the
development spreads. 0.10 was chosen because the production baseline
already sits there: its trump term contributes 0.30 × 0.928 = 0.278 of
spread against material's 3.748, i.e. **7.4%**. So the constant restates
the baseline's own balance rather than asserting something new about how
much trumps matter.

That is one free parameter, and it is reported rather than hidden.
**Search space: one dimension, three points, two model forms — six
configurations, all listed.** Selection on validation; test opened once,
afterwards.

| configuration | differs vs baseline | differs vs share 0.10 |
|---|---|---|
| structural_005 | 16.7% | **0.0%** |
| structural (0.10) | 16.7% | — |
| structural_020 | 18.3% | 3.3% |
| flexible_005 | 20.0% | **0.0%** |
| flexible (0.10) | 20.0% | — |
| flexible_020 | 20.0% | 5.8% |

Halving the constant changes **no decision at all**; doubling it changes
3–6%. The one free parameter is nearly inert across the range tested,
which is the robustness property worth having.

## 7. A sign error the methodology caught

`terminal_proximity` was written as
`1/(1+opponent_cards) − 1/(1+own_cards)`. That is **backwards**: a small
own hand means *I* am close to going out, which is good for me, so the
term must rise as my hand shrinks. As written it rewarded the opponent
being close to out.

It was not caught by review, and it survived the entire first
development measurement round. It was caught because mutation **M22**
("reverse terminal proximity's perspective") *survived* the test suite —
and when I wrote the direction test that M22 demanded, the test failed
against production, not against the mutation. The mutation was the
correct version.

The term was fixed, every affected measurement was re-run, and the
held-out test had not yet been opened, so the protocol stayed intact.
Post-fix results are the ones reported here. The oracle-domain numbers
turned out identical before and after — consistent with the single-
feature study, where `terminal_proximity` scored 68.2%, below the 83.3%
tie-break floor.

## 8. Decision quality against the exact solver

Depth 3, independent Phase 7.3.8 alpha-beta solver, positions with ≥2
legal moves that were exactly solved.

**Development (n=71):**

| evaluator | top-1 | top-2 | mean loss | worst regret | Kendall τ |
|---|---|---|---|---|---|
| baseline | 93.0% | 98.6% | 0.127 | 2.000 | 0.232 |
| mechanistic | 93.0% | 98.6% | 0.127 | 2.000 | 0.232 |
| structural | 93.0% | 98.6% | 0.127 | 2.000 | 0.230 |
| **flexible** | **94.4%** | **100.0%** | **0.099** | 2.000 | 0.234 |

**Validation (n=66) — where the model comparison happens:**

| evaluator | top-1 | top-2 | mean loss | worst regret | Kendall τ |
|---|---|---|---|---|---|
| baseline | 93.9% | 97.0% | 0.121 | 2.000 | 0.100 |
| mechanistic | 93.9% | 97.0% | 0.121 | 2.000 | 0.100 |
| structural | 93.9% | 97.0% | 0.121 | 2.000 | 0.100 |
| flexible | 93.9% | 97.0% | 0.121 | 2.000 | 0.098 |

**Held-out test (n=64), opened once, immutable:**

| evaluator | top-1 | top-2 | mean loss | worst regret | Kendall τ |
|---|---|---|---|---|---|
| baseline | 93.8% | 93.8% | 0.125 | 2.000 | 0.020 |
| mechanistic | 93.8% | 93.8% | 0.125 | 2.000 | 0.020 |
| structural | 93.8% | 93.8% | 0.125 | 2.000 | 0.020 |
| flexible | 93.8% | 93.8% | 0.125 | 2.000 | 0.026 |

E3's development advantage is **one position out of 71** (66 → 67). It
does not appear on validation, where all four are identical and E3's rank
correlation is marginally *worse*. It does not appear on the held-out
test either. This is what a validation split is for, and it did its job
before the test set was touched.

## 9. Search-level behaviour (development, 140 positions)

Root decisions differing from the baseline:

| depth | mechanistic | structural | flexible |
|---|---|---|---|
| 0 (static, no search) | 17.9% | 31.4% | 47.1% |
| 1 | 17.9% | 31.4% | 47.1% |
| 2 | 12.9% | 20.7% | 30.7% |
| 3 | 14.3% | 16.4% | 18.6% |
| 4 | 7.1% | 19.3% | 16.4% |
| 5 | 7.1% | 9.3% | 11.4% |

Every candidate is far from inert — E3 changes nearly half of purely
static decisions — and search progressively washes the differences out,
without eliminating them. Changing decisions is not improving them, and
§8 is where the quality question was actually asked.

## 10. ISMCTS (development, 30 positions, common random numbers)

| iterations | evaluator | differs vs baseline | TVD mean | TVD p95 | mean top Q |
|---|---|---|---|---|---|
| 50 | baseline | 0.0% | 0.000 | 0.000 | 341.17 |
| 50 | mechanistic | 30.0% | 0.158 | 0.380 | 340.78 |
| 50 | structural | 40.0% | 0.225 | 0.540 | 223.62 |
| 50 | flexible | 33.3% | 0.207 | 0.540 | 223.80 |
| 200 | baseline | 0.0% | 0.000 | 0.000 | 689.03 |
| 200 | mechanistic | 16.7% | 0.211 | 0.615 | 688.61 |
| 200 | structural | 26.7% | 0.265 | 0.770 | 665.10 |
| 200 | flexible | 36.7% | 0.278 | 0.860 | 665.24 |

The baseline-against-itself row is the control: 0.0% and TVD exactly
0.000 confirm the common-random-numbers setup is sound before any
difference is read from the other rows. n=30 per cell — indicative only,
and action frequency is not strength.

## 11. The frozen ply defect cannot bias this comparison

Phase 7.3.9 recorded that ISMCTS evaluates an expanding iteration's leaf
before incrementing depth, so one iteration scores a terminal at ply 0
and later ones at ply 1. The brief asks whether that can distort
evaluator comparison before any ISMCTS result is interpreted.

It cannot, and the reason is structural: the defect acts only where
`terminal_score` runs, and **no evaluator is consulted at a terminal
leaf**. It could still bias things indirectly if one evaluator steered
the search into terminals more often. Measured (30 positions, 200
iterations, common random numbers):

| evaluator | terminal leaves | per position |
|---|---|---|
| baseline | 0 | 0.00 |
| mechanistic | 0 | 0.00 |
| structural | 0 | 0.00 |
| flexible | 0 | 0.00 |

Zero terminal exposure for all four. The defect has no channel through
which to favour any evaluator here. It remains unfixed, as the phase
requires.

## 12. Cost

Measured on the evaluator directly, not through a search, over 200
distinct nodes × 5 repetitions:

| evaluator | calls/sec | mean µs | p95 µs | vs baseline |
|---|---|---|---|---|
| baseline | 666,321 | 1.34 | 1.82 | 1.00× |
| mechanistic | 577,318 | 1.59 | 2.12 | 1.15× |
| structural | 408,397 | 2.31 | 2.51 | 1.63× |
| flexible | 199,433 | 4.87 | 8.78 | **3.34×** |

E3 costs 3.3× the baseline per call. For an evaluator called millions of
times that is a real price, and it buys nothing measurable on held-out
data — which is a second, independent reason not to propose it.

## 13. Generalization

The brief requires improvement across held-out structural categories
before acceptance. There is no improvement to distribute: on validation
and on test, all four evaluators produce identical top-1, top-2, mean
loss and worst-case regret. The only candidate with any development-split
edge (E3, one position) shows none on either held-out split, in any
bucket the oracle can reach.

The deeper constraint remains the one pass 1 established: the exact
solver reaches 64 of 500 held-out positions and **63 of those are
empty-talon**, which is the region where E1 is identical to E0 by
construction. Structural generalization buckets cannot be populated from
a domain that thin.

## 14. Decision

**B — BASELINE REMAINS BEST.** Candidates were designed from game
semantics, kept information-safe, and tested; none demonstrated reliable
held-out improvement. `BaselineEvaluator` remains the production default.
All four candidates and the six sweep configurations stay registered and
selectable by name, so the work is reproducible and a later phase can
build on it without repeating it.

## 15. Limitations

- 1,500 positions from 42 trajectories; oracle coverage 64–71 per split
  and almost entirely empty-talon.
- ISMCTS cells are n=30 and indicative only.
- Deterministic stages use one true-world determinization per position to
  isolate the evaluator. That is attribution, not a claim about play
  under uncertainty.
- The antisymmetry study uses fully known positions, which the engine
  never actually searches; it characterises the evaluator, not the search.
- `AUXILIARY_SHARE` is one free parameter. It was not fitted to an
  outcome, but it was chosen, and halving or doubling it was the only
  sensitivity probe run.
- No self-play, no tournament, no playing-strength measurement of any
  kind.
