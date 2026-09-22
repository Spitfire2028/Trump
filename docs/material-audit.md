# material_persistent archaeology and causal audit — Phase 7.3.13→14

Phase 7.3.13 found that M1 is flat over ~40.8% of reachable shedding
transitions and that `trump_quality`'s apparent pathology lives in that
flat region. That created a hypothesis about M1. This phase tests it.

**Production is byte-identical to the Phase 7.3.13 package.** No repair
was implemented, because the evidence does not justify one. That is the
phase's result, not a shortfall of it.

Reproduce with:

```
python benchmarks/material_audit.py map
python benchmarks/material_audit.py flatness 0 25
python benchmarks/material_audit.py flatness 100 125
python benchmarks/material_audit.py decompose
python benchmarks/material_audit.py transience 25
python benchmarks/material_audit.py refill
python benchmarks/material_audit.py interaction
```

## 1. Source-of-truth

| component | value |
|---|---|
| formula | `talon==0 → opp − own; else max(0,opp−refill) − max(0,own−refill)` |
| weight | `material = 1.0` |
| gate | `talon_size == 0` selects the raw branch |
| normalisation | none; card units throughout |
| reference quantity | `node.refill_to` |
| `refill_to` source | `state.rules.hand_size` = **6**, a RuleSet constant |
| state-dependent? | **no** — fixed for the whole game, never changes after a move |
| rule or heuristic? | a game-rule quantity |
| consumers | `MechanisticEvaluator.evaluate` and every subclass: E2, E3, R1–R4 |

## 2. The semantic quantity, taken from the source

The class docstring states the intent rather than leaving it to be
inferred from behaviour:

> At bout resolution, while the talon still holds cards, `_refill` tops
> each hand back up to `rules.hand_size` but never removes a card. A
> deficit below that size is therefore *transient* — the refill fills it
> back in — while a surplus above it is permanent until the cards are
> played.

So:

* **material** = cards held;
* **persistent** = the part that the refill will not replace;
* **`refill_to`** is used because it is the level the rule restores to;
* **surplus above `refill_to`** is the relevant quantity because only
  that part survives a resolution;
* **flatness is intentional** — a transient difference is not a
  difference, and the evaluator declines to price it.

M1 is therefore estimating **resilience after replenishment**, not
current hand value. That distinction matters for the rest of this audit:
a feature measuring persistent material has no obligation to move when a
transient quantity moves.

Critically, this makes M1 **empirically falsifiable**: it asserts that a
sub-`refill_to` deficit gets topped back up.

### The gap the frozen rule reveals

`game/rules.py::_refill`, verbatim:

```python
order = (state.attacker, state.defender)
for player in order:
    needed = state.rules.hand_size - len(hands[player])
    drawn = talon[cursor : cursor + needed]
```

The draw is **limited by what the talon holds**, and the **attacker
draws first**. M1's formula consults neither: it applies the flat rule
whenever `talon_size > 0`, including when the talon holds one card and
the deficit is four. Whether that gap matters is an empirical question,
answered in §4.

## 3. Flatness reproduced

| | seeds 0–24 | seeds 100–124 (independent) |
|---|---|---|
| reachable shedding transitions | 7,180 | 7,014 |
| zero M1 delta | **2,926 (40.8%)** | **2,993 (42.7%)** |
| mean delta | +0.5925 | +0.5733 |
| median delta | +1.0000 | +1.0000 |

By hand size before the move (seeds 0–24) — the structure is the whole
story:

| own hand | n | flat | mean delta |
|---|---|---|---|
| 1 | 27 | 0.0% | +1.000 |
| 2 | 66 | 1.5% | +0.985 |
| 3 | 130 | 10.0% | +0.900 |
| 4 | 241 | 31.5% | +0.685 |
| 5 | 542 | 64.8% | +0.352 |
| **6** | **2,754** | **90.2%** | +0.098 |
| **7 and above** | 2,520 | **0.0%** | **+1.000** |

Above `refill_to` M1 always credits shedding. At or below it, M1 is
usually silent. By seat: attacker 44.9% flat, defender 26.9%. By card:
non-trump 40.4%, trump 41.8% — M1 does not distinguish trumps at all,
which is correct for a *material* term.

## 4. Flat-case decomposition, and the decisive test

The flat region has exactly **one** cause — it is not a grab-bag:

| flat because | n |
|---|---|
| hand at or below `refill_to`, talon alive | **2,926** |
| hand above `refill_to`, talon alive | **0** |
| talon empty | **0** |

And the converse, over the 4,254 transitions where M1 *did* move: 1,609
were the empty-talon raw branch (correct by M2) and 723 crossed
`refill_to` (correct by M1). **No case was found where M1 moved although
its semantics say it should have stayed invariant.**

### Does the transience claim actually hold?

M1 asserts sub-`refill_to` deficits are transient. That is testable by
following real games forward to see whether the hand came back up.
Reachable positions where a player was below `refill_to` with a
non-empty talon:

| | seeds 0–24 | seeds 100–124 |
|---|---|---|
| n | 584 | 564 |
| deficit later **restored** | **569 (97.4%)** | **556 (98.6%)** |
| deficit never restored | 15 (2.6%) | 8 (1.4%) |
| mean unrestored deficit | 2.07 cards | — |

Split by talon size at the moment of the deficit:

| talon | seeds 0–24 | seeds 100–124 |
|---|---|---|
| **6 or more** | **100.0% restored** (n=462) | **100.0% restored** (n=446) |
| 1–5 | 87.7% restored (n=122) | 93.2% restored (n=118) |

**M1's central empirical claim is true.** Whenever the talon holds at
least six cards it is true *without exception* on both seed blocks, and
overall it holds in 97–99% of reachable cases. The failures are confined
to the narrow talon 1–5 band, exactly where the frozen rule's talon
limit bites — the gap §2 predicted from the source.

## 5. Falsification attempt

The audit was built to be able to prove M1 wrong. Both directions were
tested.

**Against M1.** In 15 of 584 reachable cases (2.6%; 8 of 564, 1.4%, on
independent seeds) a sub-`refill_to` deficit averaging 2.07 cards was
never restored, and M1 priced it at zero. These are reachable, legal and
reproducible. The docstring's "while the talon still holds cards"
over-states the rule, which tops up only as far as the talon allows.

**For M1.** The flat region has a single coherent cause; no case exists
where M1 moves when it should not; the transience claim holds
universally at talon ≥ 6 and at 97–99% overall; and M1's raw delta over
all 5,666 shedding transitions is **never negative** (0.0%) — the term
either credits shedding or says nothing, and never penalises it. For a
material feature that is the property that matters most.

The evidence against M1 is a bounded, quantified imprecision in the
*stated condition*; the formula faithfully implements the condition it
states.

## 6. `refill_to`

A RuleSet constant of 6. Not state-dependent, unchanged by any move, a
game-rule quantity rather than an evaluator heuristic; only the value 6
was observed in play. The flat region's width is therefore fixed, rather
than narrowing as the talon runs down:

| talon | attacker needs 4 | draws | still short |
|---|---|---|---|
| 0 | 4 | 0 | 4 |
| 1 | 4 | 1 | 3 |
| 3 | 4 | 3 | 1 |
| 6 | 4 | 4 | **0** |
| 12 | 4 | 4 | **0** |

This is the precise mechanism behind §4's exception, and it explains why
the failure rate is zero at talon ≥ 6 and non-zero below it.

## 7. Interaction analysis — four questions kept separate

**Feature-level semantics** (5,666 reachable shedding transitions):

| feature | mean raw Δ | mean weighted Δ | Δ<0 | Δ=0 |
|---|---|---|---|---|
| **material (M1)** | +0.5898 | +0.5898 | **0.0%** | 41.0% |
| trump_quality | −0.1230 | −0.0849 | 20.9% | 79.1% |
| obligation (unrepaired) | +0.1220 | −0.2416 | 22.8% | 0.4% |
| terminal_proximity | +0.0093 | +0.0232 | 0.0% | 77.4% |

**Total evaluator behaviour, conditional on M1** — and this is where
Phase 7.3.13's hypothesis needs correcting:

| evaluator | M1 flat: total fell | M1 moves: total fell |
|---|---|---|
| E2, obligation **unrepaired** | **89.8%** (mean −0.398) | 2.5% (+0.763) |
| E2, obligation **repaired** | **13.6%** (mean **+0.207**) | 0.0% (+1.518) |
| E3, both repairs | 19.0% (mean +0.196) | 0.3% (+1.392) |

*(n = 2,324 flat / 3,342 moving)*

The alarming 89.8% was **overwhelmingly the already-diagnosed Defect B**,
not M1. With the obligation seat gate applied — the repair Phase 7.3.13
carried forward — the same flat region drops to **13.6%**, and its mean
turns **positive**: the evaluator still prefers shedding on average even
where M1 is silent. The residual is `trump_quality` pricing a spent
trump, which Phase 7.3.13 established as legitimate.

So M1's flatness does not produce a pathology. It removes a
counterweight, and a separate, independently identified defect then
dominated. Repair that defect and the flat region behaves acceptably.

## 8. Why no repair

Phase 7.3.14 §8 admits a repair only if all seven conditions hold.
Condition 5 — *the issue cannot be explained as an intentional
invariant* — **fails**. The flatness is the intended invariant, stated
in the source before any measurement, and it is empirically justified in
97–99% of reachable cases and universally at talon ≥ 6. Condition 6 also
weakens on the evidence of §7: most of the apparent damage was another
term's.

The remaining talon 1–5 exception is real but small, and repairing it
would mean making `refill_to` effectively talon-aware — changing the
semantic reference quantity of a validated mechanism to address 1.4–2.6%
of a sub-population. Nothing in this audit justifies that, and §9 of the
brief forbids reaching for it because shed rate would improve.

**A correct material term that is silent about transient differences is
behaving as designed.** That is what the measurements show.

## 9. What remains open

1. **The docstring over-states the rule.** "While the talon still holds
   cards, `_refill` tops each hand back up" should read "tops each hand
   back up as far as the talon allows, attacker first". This is a
   documentation imprecision with a measured consequence of 1.4–2.6% in
   a sub-population — worth correcting when M1 is next touched, not
   worth a production change on its own.
2. **Talon-aware persistence is a hypothesis, not a defect.** Whether
   `max(0, n − min(refill_to, own_cards + talon_share))` would be better
   is untested, and testing it would require the full ladder plus ISMCTS
   validation that this phase's evidence does not justify commissioning.
3. **Draw order is unmodelled.** The attacker draws first when the talon
   is short; M1 treats both seats identically. The seat asymmetry in
   flatness (attacker 44.9%, defender 26.9%) is a hand-size artifact,
   not evidence of this — but it remains unexamined.
