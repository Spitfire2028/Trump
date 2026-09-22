# ISMCTS

Single-observer Information Set Monte Carlo Tree Search, with root
re-determinization. Phase 7.3.5.

> This phase establishes architectural correctness. **No playing-strength
> claim is made or measured.** ISMCTS is not integrated into `SearchBot`.

## Why PIMC was rejected

Phase 7.3.3 measured the defect rather than assuming it. The 7.3.2
pipeline samples a world, solves it perfectly, and aggregates. Its root
criterion `max_a E[V(a,w)]` is a valid one-shot Bayesian decision — but
`V(a,w)` is itself computed assuming the player will *still know which
world they are in* at every future node.

The reproducible consequence:

```
observer P1 : 8♦ 8♠ 10♣ J♦ Q♣ K♠   trump ♠
root move   : attack 8♦
opponent    : TAKES          (reveals no card)
```

All sampled worlds land in a **provably identical** information set, yet
perfect-information search planned `attack 8♠` in some and `end attack`
in others — dump the trump eight or don't, decided by cards the player
cannot see.

## Information-set identity

`InformationSetKey = (owner, InformationSet.key())`.

The view key covers own hand, table, trump, talon **size**, hand
**sizes**, discard, attacker, phase, attack limit, the cumulative
observed set, and the durak. It contains no opponent card, no talon
contents and no allocation — which is exactly why two indistinguishable
worlds reach the same node.

`SearchInformation.fingerprint()` was rejected for identity: it embeds
marginals and allocation counts, so identical decision points would key
apart merely because a sampler differed.

## Nodes and sharing

One node per key, held in a `NodeRegistry`. Worlds reaching the same key
are handed the *same object*, so there is no code path by which they
could accumulate separate policies:

```
world A ─┐
world B ─┼──► InformationSetNode K ──► shared N, W, Q ──► one policy
world C ─┘
```

Statistics: `N(s)`, `N(s,a)`, `W(s,a)`, `Q(s,a) = W/N`, all from the
**root player's** perspective, matching the frozen evaluator and terminal
conventions. Edges are keyed by `canonical_move_key`, never object
identity.

**Nodes exist only for the observer's decisions.** That is not just a
simplification: at an observer's decision the legal moves depend on their
own hand and the public table, so they are world-independent (2,719 of
2,719 positions measured). At an opponent's decision they are not. Keeping
nodes only where the action set is world-independent means an edge means
the same thing to every world reaching it, and the "legal in one world,
not another" problem never arises at a node.

## Determinization

`Determinizer` wraps the verified Phase 7.2 `WorldGenerator`. **No second
probability model**: sampling stays uniform over compatible allocations
and every world is already validated. It exists as a named seam so the
search has exactly one place where hidden state enters — which makes the
leakage audit a matter of checking one call site.

**Re-determinization is once per iteration, at the root.** Not per node.
Per-node resampling would require rebuilding belief from a hypothesis,
and a hypothesis carries no history.

## History and the carried epistemic state

A `DeterminizedPosition` deliberately has no event log, so:

- `CardTracker.update()` **cannot** be fed a hypothetical view. It does
  not degrade — it raises `TrackerError: event stream shrank`. That is a
  standing regression test.
- The cumulative *observed* set would silently reset to "whatever is
  visible now" if recomputed. Since observed is part of identity, that
  would merge genuinely different decision points.

`EpistemicState` therefore carries the observed set forward by hand,
growing it monotonically by union. `retained_by_opponent` is carried too —
cheap, and the first thing an action-conditioned belief would need —
but nothing reads it yet, and the code says so.

## Opponent model, and the limitation

The modelled opponent acts **inside the determinized world with full
knowledge of it**, reusing the verified Phase 7.3.2a search as a
minimiser.

> **SO-ISMCTS limitation:** the observer's policy respects information
> sets; the opponent model does not. Opponent-side strategy fusion
> remains — the simulated opponent plays too knowledgeably. MO-ISMCTS is
> future work and is not implemented here.

## Evaluation and RNG

Leaves reuse `evaluate_position`: exact ply-adjusted terminal scores, and
otherwise the existing `BaselineEvaluator`. The evaluator sees the
determinized world, which is legitimate — it scores a *simulation*, it
does not identify a node.

RNG is injected; the global `random` module is never touched (monkey-
patched to raise in a test). Same information + seed + config + budget
gives the same result.

## Known limitations

1. Opponent-side fusion, above.
2. Worlds separate as soon as the opponent plays an observable card, so
   deep sharing is rare outside endgames. The architecture is correct;
   the *opportunity* for sharing is what the game structure allows.
3. Beliefs remain uniform over compatible allocations — nothing is
   inferred from how the opponent has played.
4. Not integrated into `SearchBot`; no strength evaluation.


## Phase 7.3.6 diagnostic findings

A measurement phase, not an implementation one. It asked which remaining
weakness actually costs decisions, and the answer was not what the two
obvious candidates suggested.

### Opponent-side strategy fusion is structurally absent

Across **1,379** decision nodes with two or more legal moves, the number
of groups of distinct actions that the other player cannot tell apart was
**0**. Durak is an observable-action game: every action either plays a
face-up card or is the unique non-card option, and `TAKE` and
`END_ATTACK` are never simultaneously legal.

The opponent's *choice* does differ across worlds — measured at 73.4% of
474 opponent decision nodes — but in **0 of 117** nodes where the
resulting observation was identical. Divergence is essentially always
observable, so the worlds separate legitimately.

Two things must be kept apart, and the Phase 7.3.5 report conflated them:

| | present? |
|---|---|
| opponent **strategy fusion** (unobservable divergence) | **no**, measured 0 |
| opponent **clairvoyance** (sees the observer's hand and talon) | **yes** |

MO-ISMCTS addresses the first, which does not occur here. The remaining
limitation is the second — an opponent that plays too well, biasing
values pessimistically. That is opponent over-strength, not fusion.

### Action-conditioned beliefs shift the posterior but not the decision

Opponent actions are highly informative in principle: where the opponent
took, they *also had a legal defence* in a mean of **88%** of compatible
worlds (median 100%; >50% in 109 of 120 positions). The uniform model
gives those worlds full weight, which is plainly wrong.

Yet applying an extreme upper-bound condition — zero weight to every
world in which a defence was available, discarding ~79% of them — changed
the root move in **0 of 90** decision points. Value differences were
small: mean 0.28, 95th percentile 0.88, maximum 1.01. The bound is not a
proposed opponent model; it brackets the maximum effect any
action-conditioned belief could have.

### What does move decisions

| input varied | root move changes |
|---|---|
| search depth 2 vs 5 | **29.1%** |
| extreme belief reweighting | 6.4% |
| a different world sample | 6.4% |

Evaluation horizon dominates belief quality by roughly 4.5x. Both
imperfect-information refinements are being absorbed by a crude
evaluator, so improving either first would be optimising an input the
decision is not sensitive to.

### Cost

| position | determinize | position | opponent move | key | epistemic | 100 iterations |
|---|---|---|---|---|---|---|
| early | 34.6 us | 77.7 us | 45.0 us | 15.2 us | 11.7 us | 57.2 ms |
| mid | 23.3 us | 56.9 us | 99.4 us | 22.6 us | 12.8 us | 70.7 ms |
| late | 10.9 us | 41.0 us | 100.9 us | 24.4 us | 12.3 us | 156.9 ms |
