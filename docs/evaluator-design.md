# Principled evaluator design & validation — Phase 7.3.10

What information should a non-terminal evaluator represent, what does a
candidate built only from justified components look like, and does it
survive held-out validation?

This is an **evaluator-design and validation** study. It contains no
playing-strength measurement and makes no playing-strength claim. No
coefficient was fitted to any dataset.

Reproduce with:

```
python benchmarks/evaluator_corpus.py
python benchmarks/evaluator_validation.py static   --split dev
python benchmarks/evaluator_validation.py exact    --split dev
python benchmarks/evaluator_validation.py search   --split dev
python benchmarks/evaluator_validation.py ablation --split dev
python benchmarks/evaluator_validation.py strata   --split dev
python benchmarks/evaluator_validation.py ismcts   --split dev
python benchmarks/mutate_phase7310.py
python -m pytest tests/test_evaluator.py
```

Held-out numbers below were produced by the same commands with
`--split test`, once, after the candidate was frozen.

---

## PASS 1 (material mechanism only)

The sections below are the first pass. The full-brief pass that follows
it — feature archaeology across all eight categories, the candidate
family, antisymmetry, the share sweep, and the held-out decision — is in
`docs/evaluator-design-pass2.md`.

## 1. Feature inventory

Everything an evaluator *could* read, classified before anything was
built. The evaluator receives a `SearchNode` and nothing else, so the
inventory is exactly that node's surface.

| feature | source | information class | in candidate |
|---|---|---|---|
| own hand, card by card | `node.hand` | OWN_PRIVATE | yes (trump count) |
| own unknown-identity cards | `node.unknown_own` | OWN_PRIVATE (count) | yes (via `own_cards`) |
| opponent hand **size** | `node.opponent_cards` | PUBLIC | yes |
| talon **size** | `node.talon_size` | PUBLIC | yes |
| refill size | `node.refill_to` | PUBLIC (rules) | yes |
| trump suit | `node.trump` | PUBLIC | yes |
| table slots, defended flags | `node.table` | PUBLIC | yes |
| ranks visible on the table | `node.known_ranks` | PUBLIC | no |
| attack limit | `node.attack_limit` | PUBLIC | no |
| seat (attacker/defender) | `node.searcher_is_attacker` | PUBLIC | yes |
| phase | `node.phase` | PUBLIC | yes |
| opponent hand **contents** | — | HIDDEN | **does not exist on the node** |
| talon **contents** or order | — | HIDDEN | **does not exist on the node** |
| discard pile contents | — | HIDDEN-ish | **does not exist on the node** |

The last three rows are the point of the table. The candidate does not
*decline* to read hidden information; the node it is handed has no field
that could carry any. That is a structural guarantee, and Phase 7.3.10
verified it can still fail loudly: mutation M16 adds a talon-contents
field to the node, fills it at the materialisation boundary and evaluates
on it, and the audit tests kill it.

## 2. Mechanism derivation

Candidate components were admitted only with a derivation from the frozen
Phase 2 rules. Two mechanisms survived.

**M1 — refill erasure.** At bout resolution, while the talon still holds
cards, `_refill` tops each hand back up to `rules.hand_size` and never
removes a card. A deficit below that size is therefore *transient*: the
refill fills it back in. A surplus above it is permanent until those
cards are played. So while the talon lasts, the persistent material
quantity is `max(0, n - refill_to)`, not `n`.

I first stated this as "hands are pinned at or below six while the talon
lasts", which is false, and measurement over 4,629 talon-bearing
positions from real play disproved it: 73.8% had at least one player
*above* the refill size (hand sizes 7: 519, 8: 452). Restated correctly —
only the surplus above `refill_to` persists — the term is live in most
talon-bearing positions rather than a corner case.

**M2 — terminal race.** Once the talon is empty no refill happens,
`_finish_or_start_bout` declares out any player with an empty hand, and
the player left holding cards is the durak. Every card must then be shed,
so raw `opponent_cards - own_cards` becomes exactly the right quantity.

The switch at talon exhaustion is a genuine discontinuity in the rules —
refill stops — not a smoothing parameter.

Rejected for lack of derivation: hand "quality" heuristics, trapped-card
notions, tempo, and any term requiring a view of the opponent's hand.

## 3. The candidate

`MechanisticEvaluator`, registered as `"mechanistic"`. `"baseline"`
remains the default in `SearchConfig` and `ISMCTSConfig`; nothing was
replaced.

```python
def material(self, node):
    if node.talon_size == 0:                      # M2
        return float(node.opponent_cards - node.own_cards)
    refill = node.refill_to                       # M1
    own_surplus = max(0, node.own_cards - refill)
    opponent_surplus = max(0, node.opponent_cards - refill)
    return float(opponent_surplus - own_surplus)
```

Trump bonus, obligation penalty, terminal handling and **all three
coefficients** are inherited from `BaselineEvaluator` unchanged. The only
difference between the two evaluators is the structure of the material
term, so any measured difference is attributable to the mechanism rather
than to retuning. This is deliberate: Phase 7.3.7 rejected its retuned
weights for overfitting, and this phase does not repeat the experiment.

## 4. Corpus and split

1,500 positions from legal trajectories played with the frozen Phase 2
engine, `sha256 47ea2c92abdcdbb3f8a8ed4e6909e6f5`.

| split | n | ≥2 legal moves | seeds | early / mid / late |
|---|---|---|---|---|
| development | 500 | 354 | 0–13 | 77 / 280 / 143 |
| validation | 500 | 351 | 80–92 | 72 / 276 / 152 |
| held-out test | 500 | 364 | 110–123 | 67 / 288 / 145 |

**The split is by trajectory, not by position.** Splitting positions at
random would put a position and its near-duplicate from the same game on
opposite sides, which inflates held-out agreement. Seed overlap between
splits: none.

## 5. Promotion gate — declared before any result was inspected

1. **No regression against the exact reference**: on held-out test
   positions, candidate best-move agreement not more than 2pp below
   baseline.
2. **Improvement on at least two independent dimensions**, chosen from
   exact best-move agreement, mean oracle move-value loss, and
   search-level agreement with the exact reference.
3. **No severe stratum regression**: in no stratum with n ≥ 30 is the
   candidate worse by more than 5pp.
4. **Coherent ablation**: removing the mechanism measurably changes
   behaviour.
5. **No hidden-information leakage.**
6. **No numerical pathology**: finite, bounded, deterministic.

Failing any of these means "no superior candidate", not a softened claim.

## 6. The structural result that decides the phase

**The candidate is algebraically identical to the baseline whenever
`talon_size == 0.**` With an empty talon its material term reduces to
`opponent_cards - own_cards`, which is the baseline's, and every other
term is inherited. So the two evaluators are the *same function* on
empty-talon positions.

Verified rather than asserted:

* identical on **all 143** empty-talon development positions;
* differing on **190 of 357** talon-bearing development positions;
* root decision differs in **0.0%** of empty-talon positions at depth 3,
  on development (n=38) and held-out test (n=46) alike.

Now place the independent oracle. The Phase 7.3.8 alpha-beta solver could
exactly solve 64 of the 500 held-out positions, and **63 of those 64 are
empty-talon**:

| stratum | n | baseline | mechanistic |
|---|---|---|---|
| late (empty talon) | 63 | 93.7% | 93.7% |
| mid | 1 | 100.0% | 100.0% |

The oracle's reachable domain is almost exactly the region where the
candidate is identical to the baseline by construction. This is Phase
7.3.8's "the validation domain is insufficient" sharpened into something
much more specific: **the oracle is structurally incapable of
discriminating this candidate from the baseline**, and no amount of
additional solver budget on this corpus would change that, because the
positions it can reach are the positions where there is nothing to
discriminate.

## 7. Results

### Static behaviour

| split | evaluator | min | max | mean | stdev | zeros |
|---|---|---|---|---|---|---|
| dev | baseline | −14.90 | 12.00 | 0.13 | 4.15 | 5.2% |
| dev | mechanistic | −14.90 | 12.00 | 0.15 | 3.79 | 8.0% |
| test | baseline | −18.60 | 16.50 | 0.11 | 4.06 | 4.2% |
| test | mechanistic | −18.60 | 16.50 | 0.14 | 3.73 | 6.8% |

Deterministic, finite, bounded far inside `MATE_MARGIN`. Positions where
the candidate favours the larger hand: **0** in both splits. The narrower
spread and higher zero rate are the mechanism working as derived: below
the refill size, material differences are scored as transient.

### Exact reference (depth 3)

| split | solvable | baseline agreement | mechanistic | baseline loss | mechanistic loss |
|---|---|---|---|---|---|
| dev | 71 | 93.0% | 93.0% | 0.127 | 0.127 |
| validation | 66 | 93.9% | 93.9% | 0.121 | 0.121 |
| **held-out test** | 64 | **93.8%** | **93.8%** | **0.125** | **0.125** |

Identical in every split, for the reason in §6.

### Search-level substitution (development, 180 positions)

| depth | root decision differs | mean \|ΔV\| |
|---|---|---|
| 1 | 17.8% | 0.975 |
| 3 | 15.0% | 0.641 |
| 5 | 8.9% | 0.750 |

The candidate is not inert: it changes real decisions at rates comparable
to the Phase 7.3.9 probes. The oracle simply cannot see those positions.

### Stratified decision difference (depth 3, 180 positions)

| stratum | dev n | dev differs | test n | test differs |
|---|---|---|---|---|
| 2–3 legal moves | 105 | 2.9% | 85 | 4.7% |
| 4–6 legal moves | 69 | 31.9% | 82 | 34.1% |
| 7+ legal moves | 6 | 33.3% *(n<30)* | 13 | 15.4% *(n<30)* |
| early | 38 | 21.1% | 34 | 55.9% |
| mid | 104 | 18.3% | 100 | 15.0% |
| late | 38 | **0.0%** | 46 | **0.0%** |
| talon 7+ | 97 | 20.6% | 103 | 25.2% |
| talon 1–6 | 45 | 15.6% | 31 | 25.8% |
| talon empty | 38 | **0.0%** | 46 | **0.0%** |
| attacker | 107 | 17.8% | 98 | 28.6% |
| defender | 73 | 11.0% | 82 | 7.3% |

These are *difference* rates, not regression rates: outside the
empty-talon stratum there is no oracle to say which choice was better.
The two empty-talon rows are the structural identity of §6 showing up
independently at the search level. The early-phase test rate (55.9%,
n=34) is much higher than development's (21.1%, n=38); with n≈35 either
side, that gap is not something to read a mechanism into.

### ISMCTS under common random numbers (30 positions)

| split | iterations | move differs | TVD mean |
|---|---|---|---|
| dev | 50 | 30.0% | 0.158 |
| dev | 200 | 16.7% | 0.211 |
| test | 50 | 33.3% | 0.167 |
| test | 200 | 33.3% | 0.208 |

Same seed, same budget, same determinization stream; only the evaluator
differs. Consistent with Phase 7.3.9's finding that ISMCTS is more
evaluator-sensitive than deterministic search. n=30 per cell: indicative
only, and the dev/test disagreement at 200 iterations is well within what
that n permits.

### Ablation (development, 140 positions, depth 3)

| variant | decision differs vs. full candidate |
|---|---|
| material term removed | 18.6% |
| material replaced by raw card advantage | 14.3% |
| trump term removed | 17.1% |

No component is inert. In particular the 14.3% row isolates the
mechanism itself: replacing the persistence-aware material with raw card
advantage — changing nothing else — moves one decision in seven.

## 8. Mutation audit

`src/durakfish/ai/evaluation.py` was modified by this phase, so a
mutation audit is mandatory. Seventeen mutations across the eight
required families, run on scratch copies (`benchmarks/mutate_phase7310.py`).

| family | caught |
|---|---|
| sign inversion | 2/2 |
| feature removal | 2/2 |
| feature sign reversal | 2/2 |
| weight corruption | 2/2 |
| terminal handling | 3/3 |
| perspective reversal | 2/2 |
| silent baseline fallback | 2/2 |
| hidden-information access | 1/1 |
| production default change | 1/1 |
| **total** | **17/17** |

Production source integrity verified by SHA-256 over all 41 files in
`src/` before and after: **intact**.

**The first run caught only 15 of 17, and that was the useful part.** Two
terminal-handling mutations survived: the baseline heuristic-scoring a
proven terminal whenever `ply > 0`, and the candidate returning a flat
±500 at every terminal. Both survived because the suite tested
`terminal_score()` in isolation and never checked that an evaluator
actually *routes* a decided node through it.
`test_a_decided_node_is_scored_exactly_at_every_ply` now does, and both
mutations die. A mutation that survives is a test that was missing.

## 9. Gate evaluation

| # | criterion | verdict |
|---|---|---|
| 1 | no regression > 2pp on held-out exact reference | **PASS** — 0.0pp, identical |
| 2 | improvement on ≥ 2 independent dimensions | **FAIL** — zero improvement on any oracle dimension |
| 3 | no stratum regression > 5pp at n ≥ 30 | **PASS, vacuously** — the only oracle stratum with n ≥ 30 is the empty-talon one, where the evaluators are the same function |
| 4 | coherent ablation | **PASS** — 14.3–18.6%, nothing inert |
| 5 | no hidden-information leakage | **PASS** — structural, and M16 confirms the audit is live |
| 6 | no numerical pathology | **PASS** — deterministic, bounded, 0 larger-hand preferences |

Criterion 2 fails, so the gate fails. The gate was declared before any
result was seen and is applied as written.

## 10. Conclusion

**No superior candidate.**

The candidate is not shown to be worse — on every dimension the
independent oracle can measure, it is exactly equal. It is not inert
either: it changes 15% of depth-3 root decisions and up to a third of
ISMCTS decisions. What this phase establishes is that **neither statement
can be turned into a quality claim with the evidence available**, because
the only independent oracle that exists for this game reaches only the
positions where the candidate and the baseline are provably the same
function.

That is a sharper negative result than Phase 7.3.8's, and it is
actionable in a way a diffuse "domain insufficient" was not: any future
evaluator work on talon-bearing positions needs an oracle that reaches
talon-bearing positions. Exact solving will not provide one at this
branching factor. Establishing what would is a question for a later
phase, not this one.

`BaselineEvaluator` remains the production default, unchanged.

## 11. Known limitations

- 1,500 positions from 42 seeded trajectories. Oracle coverage is 64–71
  positions per split, and almost entirely empty-talon.
- The ISMCTS stage uses 30 positions per cell. Indicative only.
- The deterministic stages evaluate one determinized world per position,
  so the evaluator is the only variable. That is a design choice for
  attribution, not a claim about play under uncertainty.
- Strata outside the empty-talon bucket report *difference*, not
  regression. No oracle exists there; nothing in this document should be
  read as saying which evaluator chose better in those positions.
- No self-play, no tournament, no playing-strength measurement of any
  kind was performed, and none is implied.
