# Evaluator incentive repair — Phase 7.3.12

Phase 7.3.11 found, in play rather than in analysis, that two features
added in Phase 7.3.10 reward behaviour opposed to the objective of
Durak. This phase reproduces both defects as arithmetic, defines what
the features should mean, repairs them, and asks whether the repairs
change behaviour in the predicted direction.

Reproduce with:

```
python benchmarks/incentive_repair.py reproduce
python benchmarks/incentive_repair.py counterfactual
python benchmarks/incentive_repair.py shed 1
python benchmarks/incentive_repair.py distribution
python benchmarks/incentive_repair.py oracle
python benchmarks/incentive_repair.py performance
python benchmarks/crossplay.py repair_pilot
python benchmarks/crossplay.py repairs 400
python benchmarks/crossplay.py analyse crossplay_repairs.json
python benchmarks/mutate_phase7312.py
```

## 1. Reproduction first — and it corrected the diagnosis

### Defect A: throw-in options, confirmed as reported

`FlexibleEvaluator._throw_in_options` counts hand cards whose rank is on
the table, weighted +0.48 for the attacker. Playing such a card removes
it from the hand, so the count falls:

```
seed 0, attacker P0
  own hand          5 -> 4      (a card was played)
  throw_in_options  1.0 -> 0.0  (weight +0.48)
  contribution      +0.480 -> +0.000   = -0.480
  full evaluation   +0.084 -> -0.990
```

**1,310** such positions in 60 seeded games. The feature charges the
searcher for exercising the option it is supposed to value.

### Defect B: **the Phase 7.3.11 diagnosis was wrong**

Phase 7.3.11 — my own report — named the fault as the denominator:
`open_attacks / own_cards` rising when the hand shrinks. Reproducing it
first, as this phase requires, found that case in **0 real positions**.
Playing a card always changes the open-attack count at the same time, so
the pure denominator effect never arises within a single ply. The
arithmetic is real in isolation:

```
1 open attack, hand 3 -> 2: pressure 0.3333 -> 0.5000, score -0.3300
```

but no legal move produces it.

Decomposing the evaluation change over 7,180 real shedding moves found
the actual culprit:

| term | mean change | negative in |
|---|---|---|
| material | +0.593 | 0.0% |
| **obligation** | **−0.236** | **76.7%** |
| trump_quality | −0.081 | 20.5% |
| terminal proximity | +0.022 | 0.0% |
| throw-in | +0.215 | 14.5% |

and splitting the obligation term by seat located it exactly:

| seat | n | mean term change | negative | baseline's gate satisfied |
|---|---|---|---|---|
| **attacker** | 5,532 | **−0.387** | **99.5%** | **0 of 5,532** |
| defender | 1,648 | +0.271 | 0.0% | 1,648 of 1,648 |

**The real defect is the missing seat gate.** `BaselineEvaluator` applies
its obligation penalty only through `_facing_attacks`. Phase 7.3.10
removed that guard because Phase 7.3.9 had measured it as satisfied at 0
of 7,591 ISMCTS leaves — a sound reason with an unchecked consequence.
Ungated, the term counts *every* undefended slot against the searcher,
including the attack the searcher has just made. That card is the
opponent's problem. Attacking is punished 99.5% of the time.

This is larger than defect A and has a different cause than reported.
Reproducing before repairing is what caught it.

## 2. Intended semantics

**Throw-in.** The rules permit an addition only at a rank already in
play, so what carries attacking value is the *breadth of ranks* the hand
can still contribute to — not how many duplicates are hoarded against
them. Holding three sixes is one rank of continuation, not three.

```
attack_continuation = |{ r in known_ranks : some card in hand has rank r }|
```

Playing one of two matching cards leaves this unchanged: the rank stays
on the table and the hand still covers it. Playing the last card of a
rank does reduce it — a genuine loss of continuation, not an artifact,
and the documented exception to monotonicity.

**Obligation.** An undefended attack is the searcher's obligation exactly
when the searcher is the bout's defender. Gate on seat, and on seat only:
what Phase 7.3.9 found unreachable at ISMCTS leaves was the *phase*
condition, not the seat condition, so this keeps the term live where the
baseline's version is dead. Normalisation is dropped — dividing by a
quantity the player is trying to shrink was never motivated, and with the
gate in place there is nothing left for it to fix.

## 3. The repair family

| | definition | base class |
|---|---|---|
| **R0** `flexible` | unrepaired Phase 7.3.10 candidate | — |
| **R1** `flex_r1` | rank-coverage throw-in only | E3 |
| **R2** `flex_r2` | seat-gated obligation only | E2 |
| **R3** `flex_r3` | both repairs | E3 |
| **R4** `flex_r4` | R3 with obligation ÷ `attack_limit` | E3 |

No coefficient was changed. Every weight is the Phase 7.3.10
auxiliary-share value, so any difference is attributable to the semantic
repairs alone.

**One caveat on the comparison.** R2 is built on E2 and so has no
flexibility terms at all, while R1/R3/R4 are built on E3. The controlled
ladder that isolates each repair is therefore **R0 → R1 → R3**, all on
E3. R2 shows the obligation repair working on the E2 line as well, but
R2-vs-R3 mixes the throw-in repair with the presence of
`defense_flexibility` and should not be read as isolating either.

## 4. Counterfactual incentive tests

Over 7,180 real shedding moves: how often does the evaluator prefer the
position *before* the card was played?

| evaluator | prefers pre-shed | mean change |
|---|---|---|
| baseline | **0.0%** | +1.046 |
| mechanistic | 6.0% | +0.639 |
| structural (E2) | 38.0% | +0.298 |
| **R0** flexible | **38.1%** | +0.260 |
| **R1** throw-in repair | **38.1%** | +0.223 |
| **R2** obligation repair | **5.3%** | +0.988 |
| **R3** both | **7.7%** | +0.914 |
| **R4** bounded | 18.6% | +0.537 |

R1 alone changes nothing — the obligation defect dominates so completely
that repairing the throw-in is invisible underneath it. R2 collapses the
rate from 38% to 5%.

## 5. Behaviour: does the policy shed?

The measurement that exposed the defect in Phase 7.3.11, repeated for
every repair. Identical positions from real games, depth 1, n = 1,135
choices:

| evaluator | sheds a card |
|---|---|
| baseline | 97.4% |
| structural (E2) | 74.9% |
| **R0** flexible | **53.7%** |
| **R1** | **53.7%** |
| **R2** | 91.4% |
| **R3** | **92.2%** |
| **R4** | 66.0% |

And the throw-in repair on its own terms — does shedding still reduce the
attack-continuation feature? Over 5,532 attacker shedding moves:

| feature | feature falls |
|---|---|
| R0 `throw_in_options` | 14.5% |
| R1 rank coverage | **8.9%** |

A 39% relative reduction. The residual is the documented exception:
playing the last card of a rank is a real loss of continuation.

## 6. Self-play revalidation — 6,000 seed-pairs, 12,000 games

Same harness, same paired design, same declared 5pp threshold as Phase
7.3.11. Pilot sd 0.281–0.340 → up to 364 pairs needed; **400 per cell**
were run. Score is the **baseline's**, so lower means the candidate did
better.

| candidate | depth 1 | depth 3 | depth 5 |
|---|---|---|---|
| R0 flexible | **0.946** | 0.538 | 0.515 |
| R1 | **0.939** | 0.535 | 0.515 |
| R2 | **0.492** | 0.477 | 0.507 |
| R3 | **0.470** | 0.473 | 0.509 |
| R4 | **0.846** | 0.526 | 0.506 |

At depth 1, with 95% intervals: R0 [0.929, 0.961], R1 [0.922, 0.955],
R2 [0.463, 0.524], R3 [0.436, 0.503], R4 [0.821, 0.869].

Three results, in order of importance:

1. **The obligation repair works, and the effect is enormous.** On the
   controlled E3 ladder, R1 → R3 moves the baseline's depth-1 score from
   0.939 to 0.470 — a **47-point swing** from one seat gate.
2. **The throw-in repair changes nothing measurable.** R0 → R1 moves the
   score by 0.7pp, well inside noise. It is semantically correct and
   behaviourally negligible.
3. **The elegant normalisation fails.** R4 divides the obligation by a
   hand-independent constant, which looks more principled than R3's raw
   count, and it leaves the candidate at 0.846 — most of the defect
   intact. The reason is that the ÷6 shrinks the *defensive* reward that
   was offsetting a third hoarding pull (see below). Mathematical
   tidiness is not evidence.

**No repair beats the baseline.** R3's depth-1 score of 0.470 means the
candidate scored 0.530 — 3pp, below the declared threshold, p = 0.079.
At depth 3 and 5 every cell is equivalent, exactly as Phase 7.3.11 found:
search repairs what the evaluator gets wrong.

## 7. A third incentive defect, newly identified

The decomposition surfaced one the earlier phases did not name:
`trump_quality` is negative on **20.5%** of shedding moves, mean −0.081,
because playing a trump removes it from the hand and lowers the term. It
is the same class of error as defect A — a "holdings" feature charging
for use — and it is why R2 and R3 sit at 5–8% rather than the baseline's
0%.

It is recorded, not repaired. The brief scoped this phase to the two
demonstrated defects, and widening scope mid-phase is how an audit turns
into a redesign.

## 8. Exact-reference check

12 solvable positions at depth 3. All six evaluators: 91.7%
oracle-optimal, mean regret 0.167 — **identical**. Coverage this small
and this structurally atypical cannot discriminate the candidates, and is
not used to.

## 9. Cost

200 nodes × 5 repetitions:

| evaluator | calls/sec | mean µs | p95 µs | vs baseline |
|---|---|---|---|---|
| baseline | 803,514 | 1.13 | 1.49 | 1.00× |
| R0 flexible | 232,218 | 4.19 | 7.30 | 3.46× |
| R1 | 248,463 | 3.92 | 7.06 | 3.23× |
| **R2** | **598,986** | **1.55** | **2.29** | **1.34×** |
| R3 | 262,643 | 3.69 | 7.19 | 3.06× |
| R4 | 269,972 | 3.59 | 7.10 | 2.98× |

R2 is both the effective repair and by far the cheapest, because the
seat gate short-circuits for the attacker and E2 carries no flexibility
terms. Since the throw-in repair buys nothing measurable, R2 is the
better engineering object of the two — at equal strength, a third of the
cost of R3.

## 10. What this does and does not establish

It establishes that a demonstrated incentive defect can be located
precisely, repaired from semantics rather than by sign-flipping or
coefficient tuning, and that the repair changes behaviour in exactly the
predicted direction: hoarding 38% → 8%, shed rate 54% → 92%, attacker
charge → 0, depth-1 self-play 0.05 → 0.53.

It does not establish that the repaired candidate plays better than the
baseline. It does not, and 3pp at p = 0.079 is not an improvement.

The honest statement is the one the brief anticipates: *the candidate
contained a demonstrable incentive defect, correcting it recovers the
strength the defect was destroying, and it does not yet establish
superior playing strength.*
