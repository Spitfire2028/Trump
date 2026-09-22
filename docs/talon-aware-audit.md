# Talon-aware persistence and draw-order audit — Phase 7.3.15

Phase 7.3.14 left one hypothesis standing: M1 treats `refill_to = 6` as
a constant and models neither the talon's actual capacity nor the
attacker-first draw order. This phase asks whether that abstraction is
*semantically wrong*, not whether the omission exists — it does.

**Production is byte-identical to the Phase 7.3.14 package.** The
candidate models were built as scratch implementations only.

Reproduce with:

```
python benchmarks/talon_aware_audit.py boundary
python benchmarks/talon_aware_audit.py models
python benchmarks/talon_aware_audit.py exceptions 200 225
python benchmarks/talon_aware_audit.py draworder 25
python benchmarks/talon_aware_audit.py scope 25
python benchmarks/talon_aware_audit.py accuracy 25
```

## 1. Information boundary — the question that had to come first

`information/information_set.py`, line 37:

> The talon's *contents and order* are hidden; only its size is public.

The three quantities the brief insists on separating:

| quantity | status | available to M1? |
|---|---|---|
| talon **size** | PUBLIC | **yes** — `node.talon_size`, an int |
| talon **composition** | HIDDEN | **no** — no `SearchNode` field can hold cards |
| future **draw order** | split | *which player* draws first is a rule constant plus public seat → **yes**; *which cards* they draw → **no** |

So a counts-only talon-aware model is **architecturally constructible**,
using `own_cards`, `opponent_cards`, `talon_size`, `refill_to` and
`searcher_is_attacker` — every one permitted. This phase therefore
cannot reject the candidate on information grounds, and the question has
to be decided on semantics.

## 2. The competing models

**Model A — current production M1**

```
talon == 0 → opp − own
otherwise  → max(0, opp − refill) − max(0, own − refill)
```

**Model B — talon-aware, no priority.** Shortfall split proportionally
when the talon cannot cover both.

**Model C — talon-aware and attacker-first.**

```
needed(x)  = max(0, refill_to − x)
share      = attacker's need taken first, then the defender's
post_x     = x + min(needed(x), share_x)
M1_C       = post_opp − post_own
```

Model C is not an ad-hoc third formula. **It reduces to production M1 in
both of that formula's branches:** with an ample talon every hand
reaches `refill_to` and the expression becomes
`max(0,opp−refill) − max(0,own−refill)`; with an empty talon nobody
draws and it becomes `opp − own`. Model C is the single expression that
interpolates between the two branches the current M1 switches between —
the strongest form of the candidate, and the one worth testing.

Verified on 2,157 reachable positions: **618 empty-talon agree, 1,523
ample-talon agree, 16 short-talon differ.** `model_a` was also asserted
equal to `MechanisticEvaluator.material` at every position.

## 3. Talon-short exceptions, fresh seeds

Complete table, seeds 200–224: **17 unrestored-deficit cases**, mean
unrecovered deficit 1.59 cards, attacker/defender 15/2. Representative
rows:

| seed | seat | hand | opp | talon | needs | shortfall |
|---|---|---|---|---|---|---|
| 201 | attacker | 5 | 7 | 1 | 1 | **0** |
| 201 | attacker | 4 | 7 | 1 | 2 | 1 |
| 218 | attacker | 3 | 11 | 2 | 3 | 1 |
| 224 | defender | 5 | 5 | 1 | 1 | **0** |

### The shortfall column is the finding

Many rows show **shortfall 0** — the talon *did* hold enough at that
moment, yet the deficit was never restored. Those are not capacity
failures; the game ended, or the player kept playing, before the refill
arrived.

Measured over fresh seeds 200–239, of 19 unrestored-deficit cases:

| | n | share |
|---|---|---|
| talon genuinely too short (capacity failure) | **9** | **47.4%** |
| talon sufficient; unrestored for another reason | 10 | 52.6% |

**Only about half of Phase 7.3.14's exception population is a capacity
question at all.** The rest is end-of-game timing, which no talon-aware
formula would fix and which is not evidence against M1.

## 4. Draw-order audit

Over 2,719 reachable positions (seeds 0–24):

| | n | share |
|---|---|---|
| talon cannot satisfy both players | 16 | 0.6% |
| …of those, both players actually need cards | 4 | 0.15% |
| …where priority changes the allocation | **2** | **0.07%** |

Attacker-first priority changes the semantic quantity in **2 of 2,719
reachable positions**. The case requires a triple coincidence: both
players short, the talon insufficient, *and* the split contested.

That is the answer to the draw-order question. The refinement is
semantically real and empirically inert, and it is the distinctive claim
Model C makes over Model B.

## 5. Quantified scope — with denominators

Over 2,719 reachable decision positions (seeds 0–24):

| | n | share |
|---|---|---|
| hand below `refill_to` | 1,028 | 37.8% |
| talon too short to satisfy both | 16 | **0.6%** |
| **Model A differs from Model C** | **16** | **0.6%** |
| mean \|A − C\| where they differ | 1.38 | |
| max \|A − C\| | 2 | |
| shedding transitions where the A and C *deltas* differ | 30 / 5,666 | **0.5%** |

By talon size: 14 cases at talon 1, 2 at talon 5. By seat: 9 attacker,
7 defender.

## 6. The semantic test

Not shed rate, not self-play. M1 claims to estimate material surviving
replenishment, so each model is scored on how accurately it predicts
**the hand-size difference the real game actually produced at the next
refill**, on reachable play.

**All positions** (1,958 followed to an actual refill, seeds 0–24):

| model | mean \|error\| | median | exact hits |
|---|---|---|---|
| A — current | 2.2002 | 1.00 | 18.3% |
| B — talon-aware | 2.1931 | 1.00 | 18.4% |
| C — attacker-first | 2.1931 | 1.00 | 18.4% |

A 0.3% relative improvement and 0.1pp more exact hits. **No meaningful
difference.**

**Restricted to positions where the models actually disagree** (42
positions, seeds 0–59 — the fair test of whether C is better where it
matters):

| model | mean \|error\| | median | exact hits |
|---|---|---|---|
| A — current | 3.9286 | 4.00 | 7.1% |
| B — talon-aware | **3.3810** | 3.00 | 9.5% |
| C — attacker-first | 3.4762 | 3.00 | **11.9%** |

Where they differ, the talon-aware models *are* more accurate — a 12–14%
relative improvement on n=42. But **B and C are indistinguishable from
each other**, and B has the lower mean error while C has more exact
hits. With n=42 that ordering is noise, and it independently confirms
§4: the attacker-first refinement adds nothing over plain talon
awareness.

## 7. Falsification, both directions

**For a defect.** In 0.6% of reachable positions the current M1 asserts
a transience the frozen rule cannot deliver, and where it does so the
talon-aware models predict reality better. Reachable, legal,
reproducible.

**Against a defect.** The affected population is 0.6% of positions and
0.5% of shedding transitions; about half of the exception cases are
end-of-game timing rather than capacity; the aggregate predictive
accuracy gain is 0.3%, inside noise; the draw-order refinement moves 2
positions in 2,719; and the restricted advantage rests on n=42 with B
and C tied. The mean absolute difference where the models disagree is
1.38 cards against an evaluator whose material term already carries
weight 1.0 in a score whose other terms sum to a fraction of that.

The evidence for a defect is real, precisely located, and **too small to
support changing a validated mechanism.**

## 8. Interaction and search

Under the repaired evaluator (`lad_e3_both`, seat-gated obligation — the
old ungated term was deliberately not used), substituting Model C's
material changes **1.46% of root decisions at depth 1 and 0.99% at depth
3** (n = 1,512 and 1,309).

**No large search or ISMCTS experiment was commissioned, and the
arithmetic says why.** The established paired design has sd ≈ 0.28–0.32
and needs ~250–360 pairs to resolve 5pp. Detecting even a 1pp difference
would need ((1.96+0.84)×0.30/0.01)² ≈ **7,056 pairs ≈ 14,112 games per
cell**, and nothing in the semantic audit predicts any difference at
all. Running it would be effort without a hypothesis — precisely what
§14 of the brief forbids.

## 9. Information safety

Hidden-permutation invariance over 14 indistinguishable world pairs × (2
scratch evaluators + both raw models): **0 violations**. Models B and C
consume only counts; no card identity is reachable from them. Production
was never modified, so no key, cache, determinization or aggregation
path changed.

## 10. Decision

**Model A — the current M1 — is semantically adequate.** Its abstraction
is not merely defensible: the constant `refill_to` reference is correct
in 99.4% of reachable positions, and where it is wrong the improvement
available is 0.3% in aggregate and rests on 42 observations.

The candidate passes the information audit and fails the semantic one.
Per the brief's closing principle — *do not make M1 talon-aware merely
because the game engine is talon-aware* — the evaluator's abstraction is
doing its job.

## 11. Documentation

Phase 7.3.14's flagged imprecision stands. Current wording:

> while the talon still holds cards, `_refill` tops each hand back up to
> `rules.hand_size`

Accurate wording:

> at bout resolution `_refill` restores hands toward `rules.hand_size`
> **as far as the remaining talon permits, attacker first**

This materially affects future implementation correctness — someone
reading the current sentence could reasonably implement unconditional
restoration. **It is recommended but deliberately not applied here**, so
that this phase's "no production change" result stays verifiable by a
byte-level diff. Applying it is a one-line, behaviour-free edit for
whichever phase next touches M1, and it conceals nothing: the semantic
question it relates to is now answered.
