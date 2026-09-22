# Causal consolidation, ISMCTS revalidation, trump_quality audit — Phase 7.3.13

Three questions Phase 7.3.12 left open: its R2-vs-R3 comparison was
confounded, its repairs were never run through ISMCTS, and it named
`trump_quality` a third defect without proving it.

**Production is byte-identical to the Phase 7.3.12 package.** Every
variant this phase needed is registered into the existing `EVALUATORS`
registry at runtime from `benchmarks/`, the pattern Phase 7.3.9
established. No repair was implemented, because the evidence did not
justify one.

Reproduce with:

```
python benchmarks/causal_ladder.py map
python benchmarks/causal_ladder.py reproduce
python benchmarks/causal_ladder.py pilot 40
python benchmarks/causal_ladder.py play 400
python benchmarks/causal_ladder.py ismcts 40 50
python benchmarks/causal_ladder.py analyse ladder_deterministic.json
python benchmarks/causal_ladder.py analyse ladder_ismcts_50.json
python benchmarks/trump_audit.py reproduce
python benchmarks/trump_audit.py decompose
python benchmarks/trump_audit.py gates
python benchmarks/trump_audit.py legitimacy
```

## 1. Source-of-truth map

Read from source, not from any previous report. Weights are the Phase
7.3.10 auxiliary-share values, unchanged throughout.

| feature | formula | weight | gate | normalisation |
|---|---|---|---|---|
| `trump_quality` | `Σ (rank−6)/8` over own trumps | +0.69 | **none** | per-card rank |
| `obligation` (E2) | `open_attacks / own_cards` | −1.98 | **none** | hand size |
| `obligation` (repaired) | `0` if attacker else `open_attacks` | −1.98 | seat | none |
| `obligation` (R4) | `0` if attacker else `open_attacks/attack_limit` | −1.98 | seat | attack limit |
| `throw_in` (E3) | count of hand cards whose rank is in play | +0.48 | attacker | none |
| `throw_in` (repaired) | count of **distinct ranks** in play covered | +0.48 | attacker | none |
| `defense_flex` | hand cards beating every open attack | +0.29 | defender | none |

Class chain, from `__mro__`:

```
MechanisticEvaluator
  └ StructuralEvaluator (E2)         owns _trump_quality, _obligation_pressure
      ├ FlexibleEvaluator (E3)       owns _throw_in_options, _defense_flexibility
      │   └ RepairedThrowInEvaluator (R1)   owns _throw_in_options
      │       └ RepairedEvaluator (R3)      owns _obligation_pressure
      │           └ RepairedBoundedEvaluator (R4)
      └ RepairedObligationEvaluator (R2)    owns _obligation_pressure
```

## 2. The R2/R3 comparison was worse than confounded

Phase 7.3.12 noted R2 uses E2 and R3 uses E3. The `__mro__` shows why
that matters more than the report conveyed: **R2 is not "E3 with one
repair", it is a different evaluator family.**

| | base | flexibility terms | obligation | throw-in |
|---|---|---|---|---|
| R2 `flex_r2` | E2 | **none** | repaired | **n/a** |
| R3 `flex_r3` | E3 | throw-in + defence | repaired | repaired |

R2-vs-R3 varies three things at once. No claim about "which repairs
belong together" survives it — and Phase 7.3.12's recommendation to
carry R2 forward "at equal strength, a third of the cost" rested on it.

## 3. The controlled ladder

The cell Phase 7.3.12 never built is **E3 with the obligation repair and
the original per-card throw-in**. Without it the obligation repair
cannot be isolated on the E3 line.

| cell | definition |
|---|---|
| `lad_e3_none` | E3 unchanged (= `flexible`, R0) |
| `lad_e3_ti` | E3 + throw-in repair (= `flex_r1`, R1) |
| `lad_e3_ob` | **E3 + obligation repair** (new — the gap) |
| `lad_e3_both` | E3 + both (= `flex_r3`, R3) |
| `lad_e2_none` | E2 unchanged (= `structural`) |
| `lad_e2_ob` | E2 + obligation repair (= `flex_r2`, R2) |

The throw-in factor does not exist on the E2 line — E2 has no throw-in
term — so the design is a 2×2 on E3 plus a 1×2 on E2, not a 2×3.

**All six cells verified distinct** over 1,636 reachable positions, and
the distinctness pattern is itself evidence the factorial is clean:

| contrast | positions differing |
|---|---|
| throw-in repair (at either obligation level) | 83 / 1,636 |
| obligation repair (at either throw-in level) | 898 / 1,636 |
| base E2 → E3 | 760 / 1,636 |

The throw-in effect is 83 at *both* obligation levels and the obligation
effect is 898 at *both* throw-in levels — additive, no interaction.

## 4. Deterministic ladder results

7,200 seed-pairs / 14,400 games. 400 pairs per cell, paired design, both
seat orientations, deals shared exactly by seed. Pilot sd 0.124–0.321 →
≤324 pairs needed for 5pp at 80% power. Scores are the **candidate's**.

| cell | depth 1 | depth 3 | depth 5 |
|---|---|---|---|
| `lad_e3_none` | 0.054 | 0.462 | 0.485 |
| `lad_e3_ti` | 0.061 | 0.465 | 0.485 |
| `lad_e3_ob` | **0.535** | 0.527 | 0.491 |
| `lad_e3_both` | **0.530** | 0.527 | 0.491 |
| `lad_e2_none` | 0.349 | 0.507 | 0.488 |
| `lad_e2_ob` | **0.508** | 0.523 | 0.493 |

Isolated factor effects (change in candidate score):

| factor | depth 1 | depth 3 | depth 5 |
|---|---|---|---|
| throw-in repair, no obligation repair | +0.006 | +0.003 | +0.000 |
| throw-in repair, with obligation repair | −0.005 | +0.000 | +0.000 |
| **obligation repair, no throw-in repair** | **+0.481** | **+0.065** | +0.006 |
| **obligation repair, with throw-in repair** | **+0.469** | **+0.062** | +0.006 |
| base E2 → E3, both unrepaired | −0.294 | −0.045 | −0.003 |
| base E2 → E3, obligation repaired | +0.027 | +0.004 | −0.002 |

**The throw-in repair does nothing, confirmed at both levels of the
other factor and at every depth.** Its two estimates bracket zero
(+0.006 and −0.005). The obligation repair carries the entire effect,
and it is consistent whether or not the throw-in repair is present.

### What this says about R2 versus R3

R3 (0.530) minus R2 (0.508) at depth 1 is +0.022, and the ladder
decomposes it exactly: **+0.027 from the E3 base, −0.005 from the
throw-in repair.** Both below the declared 5pp threshold, individually
and together.

So the two are *close*, but the phase brief's caution is the right one:
this does **not** establish equivalence. It establishes that the
difference between them is below the threshold this project declared
practically meaningful, and that the difference is mostly base class
rather than repair.

## 5. ISMCTS revalidation

Phase 7.3.12's stated limitation — "deterministic search only" —
corrected. 160 seed-pairs / 320 games, 50 iterations, common random
numbers, the frozen ISMCTS path unmodified.

| cell | n | candidate score | 95% CI (baseline's) | p |
|---|---|---|---|---|
| `lad_e3_none` | 40 | 0.369 | [0.525, 0.738] | 0.021 |
| `lad_e3_ti` | 40 | 0.381 | [0.512, 0.725] | 0.037 |
| `lad_e3_ob` | 40 | 0.475 | [0.425, 0.625] | 0.722 |
| `lad_e3_both` | 40 | 0.488 | [0.412, 0.625] | 0.909 |

| factor | effect |
|---|---|
| throw-in repair, no obligation repair | +0.012 |
| throw-in repair, with obligation repair | +0.013 |
| obligation repair, no throw-in repair | +0.106 |
| obligation repair, with throw-in repair | +0.106 |

**The deterministic conclusions survive ISMCTS.** Same ordering, same
sign, same structure: the unrepaired cells differ significantly from the
baseline (p = 0.021, 0.037) and the obligation-repaired cells do not
(p = 0.722, 0.909); the throw-in effect is negligible at both levels.
The obligation effect is smaller here (+0.106) than at depth 1 (+0.481),
consistent with the depth gradient, since ISMCTS-50 searches deeper than
one ply.

**Honest power statement.** At n = 40 and sd ≈ 0.32 this arm resolves
about 14pp. The +0.106 factor effect is *below* that as an isolated
contrast, so it is directionally consistent with the deterministic
result rather than independently significant. The cell-level p-values
are the stronger evidence here.

## 6. trump_quality — reproduced on reachable positions

7,180 reachable legal shedding transitions from 25 seeded games.

The 20.5% figure reproduces exactly. What it is made of does not match
the Phase 7.3.12 characterisation:

| | |
|---|---|
| transitions with a negative `trump_quality` delta | 1,473 (20.5%) |
| …of those, the move actually played a trump | **1,473 (100.0%)** |
| …of those, the move played no trump | **0** |
| moves playing a non-trump | 5,540 — **0.0% negative**, mean delta exactly 0 |
| moves playing a trump | 1,640 — 89.8% negative |
| delta equals minus the played trump's normalised rank | **1,640 of 1,640** |

The remaining 10.2% of trump moves are the six of trumps, which
contributes exactly `(6−6)/8 = 0`.

## 7. trump_quality — the mechanism

Working through the brief's seven candidate explanations:

**Not a gating problem.** No gate isolates it:

| gate | applies to | negative | mean weighted delta |
|---|---|---|---|
| none (current) | 100.0% | 20.5% | −0.081 |
| defender only | 23.0% | 47.0% | −0.188 |
| attacker only | 77.0% | 12.6% | −0.049 |
| DEFENSE phase only | 23.0% | 47.0% | −0.188 |
| talon non-empty | 77.6% | 21.8% | −0.088 |

The defender rate is higher only because defenders play trumps more
often — beating an attack frequently requires one. Conditional on a
trump being played, every subset behaves identically. Unlike Defect B,
where the attacker/defender split was 99.5% / 0.0%, a gate here would
hide the term rather than fix it.

**Not the same class of error as Defect A.** `throw_in_options` was
defective because holding three sixes counted as three units of
continuation, so playing one lost a third of a capability that had not
actually changed — an accounting artifact from duplication. That cannot
happen here: one suit is trump, so **each trump rank appears at most
once in a hand**. Every trump played is the last of its rank — the audit
shows "last of rank" and "trump played" are the same 1,640 rows — and
the feature reports exactly the value of the specific card spent. Phase
7.3.12's claim that the two are "the same class of error" is **wrong**
and is corrected here.

**It is an interaction with `material_persistent`.** Ablating
`trump_quality` to zero recovers almost the whole shed-rate gap:

| evaluator | shed rate, depth 1, n = 1,135 |
|---|---|
| baseline | 97.4% |
| `flex_r2` | 91.4% |
| `flex_r2`, trump_quality = 0 | **97.1%** |
| `flex_r3` | 92.2% |
| `flex_r3`, trump_quality = 0 | **97.3%** |

So the term does suppress shedding. But the decisive measurement is
*where*:

| trump-playing moves | n | full evaluation fell |
|---|---|---|
| where `material_persistent` credits shedding | 954 | **3.6%** |
| where `material_persistent` credits nothing | 686 | **83.7%** |

`material_persistent` gives **zero** credit for shedding in **40.8%** of
reachable shedding transitions — by design, since Phase 7.3.10's
mechanism M1 counts only the surplus above `refill_to`, which is flat
while the talon lasts and the hand is at or below six.

**The diagnosis: `trump_quality` is correctly signed, exactly computed,
free of duplication artifacts and ungateable. It only outweighs shedding
where the material term has priced shedding at zero.** Two individually
well-motivated terms combine into a bad incentive; neither is wrong on
its own. In the brief's taxonomy this is explanation **#3 — an artifact
of the evaluator's other terms** — not #1, and not #5.

## 8. Why no repair was implemented

Phase 7.3.13's §10 permits a repair only when ten conditions hold. The
causal mechanism *is* established, but it implicates
`material_persistent`, which is the validated Phase 7.3.10 mechanism M1,
not `trump_quality`. Repairing the thing the evidence actually indicts
would mean reopening M1 — outside this phase's scope, and the brief
explicitly permits characterisation without repair.

Charging a player for spending a high trump is, on its face, correct
Durak. Removing that charge to raise a shed-rate statistic would be
optimising for a favourable measurement, which this phase was told not
to do.

## 9. Categorised incentive tests

Share of reachable shedding moves after which the full evaluation falls:

| category | n | e3_none | e3_ti | e3_ob | e3_both | e2_none | e2_ob |
|---|---|---|---|---|---|---|---|
| attacker moves | 5,532 | 34.0% | 34.0% | 11.5% | 9.8% | 46.7% | 6.9% |
| defender moves | 1,648 | 51.9% | 51.9% | **0.5%** | **0.5%** | 8.7% | **0.0%** |
| trump played | 1,640 | 59.7% | 59.9% | 19.1% | 19.7% | 37.1% | 23.4% |
| non-trump played | 5,540 | 31.7% | 31.6% | 6.0% | 4.1% | 38.2% | **0.0%** |

Two of the brief's six categories collapse onto others and are reported
as such rather than padded out: "last card of a trump rank" is exactly
"trump played" (1,640 rows, for the uniqueness reason above), and
"ordinary continuation" is exactly "non-trump played".

After the obligation repair, the only category with a substantial
residual is trump-playing moves. Non-trump moves sit at 0.0–6.0%. That
independently confirms the ablation: `trump_quality` is the entire
remaining signal, and it is legitimate pricing.

## 10. Information safety

Hidden-permutation invariance over 14 indistinguishable world pairs × 6
ladder cells: **0 violations**. Every cell reads only `node.hand`,
`node.table`, `node.known_ranks`, `node.searcher_is_attacker`,
`node.attack_limit` — the approved surface. No new information channel
was introduced, because no production code changed.

## 11. Performance, controlled

200 reachable nodes × 5 repetitions:

| cell | calls/sec | mean µs | p95 µs | vs baseline |
|---|---|---|---|---|
| baseline | 849,727 | 1.05 | 1.82 | 1.00× |
| `lad_e3_none` | 285,244 | 3.37 | 5.91 | 2.98× |
| `lad_e3_ti` | 276,424 | 3.46 | 6.20 | 3.07× |
| `lad_e3_ob` | 280,069 | 3.41 | 6.53 | 3.03× |
| `lad_e3_both` | 274,890 | 3.46 | 6.97 | 3.09× |
| `lad_e2_none` | 595,431 | 1.55 | 1.78 | 1.43× |
| `lad_e2_ob` | 680,015 | 1.33 | 1.65 | 1.25× |

**A third correction to Phase 7.3.12.** That phase reported R2 as
costing "a third of R3" and inferred the repair was cheap. Controlled,
the four E3 cells all cost 2.98–3.09× — the repairs are cost-neutral —
and the E2 line costs 1.25–1.43×. The cost difference is the base
class, not the repair.

## 12. Reproducibility, shown

| cell | run A | run B | identical pairs |
|---|---|---|---|
| `lad_e3_none` | 0.9417 | 0.9417 | 30/30 |
| `lad_e3_ob` | 0.3500 | 0.3500 | 30/30 |
| `lad_e3_both` | 0.4083 | 0.4083 | 30/30 |

And against the stored run: `lad_e3_ob` depth 1, first 30 seeds — saved
mean 0.3500, fresh mean 0.3500, identical in 30/30 pairs.

## 13. Three corrections to Phase 7.3.12

1. **`trump_quality` is not "the same class of error as Defect A".** No
   duplication artifact is constructible; the feature reports exactly
   the card spent. The pathology is an interaction with the material
   term's flatness.
2. **"R2 equals R3 at a third of the cost" was a confounded claim.**
   Controlled, the strength difference is +0.022 (below threshold) and
   the cost difference is entirely base class.
3. **The throw-in repair's value was overstated by omission.** Phase
   7.3.12 reported it as "semantically correct, behaviourally
   negligible"; the factorial now shows its two isolated estimates
   bracket zero at every depth and in ISMCTS, which is stronger.
