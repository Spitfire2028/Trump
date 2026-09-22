# Search

Phase 5 adds the first agent that looks ahead. This document is the
architecture, the semantics, and — most importantly — the limits.

## The obstacle, stated first

A full-width game tree **cannot** be built from an `InformationSet`. Two
things block it, and neither is an implementation shortcoming:

- **Opponent replies** need their card identities. Hidden.
- **Anything past a bout resolution** needs the talon order, because both
  players draw. Hidden.

Determinization — sampling hidden hands and searching each sampled world —
is the standard answer, and it is explicitly out of scope for this phase.
So the search does the only honest thing left: it expands the searcher's
own moves exactly, models the opponent by **action class** rather than by
card, and stops at any frontier where continuing would need knowledge it
does not have.

**Phase 5 performs no hidden-card inference.** No card tracking, no
probabilities, no enumeration of hidden hands or talons, no Bayesian
reasoning, no determinization.

## What a search node is

A `SearchNode` is not a `GameState` and not an `InformationSet`. It is:

> the searcher's own hand, plus the public position, plus a conservative
> abstraction of what the opponent may do next.

| Field | Source | |
|---|---|---|
| `hand` | own | exact cards |
| `unknown_own` | derived | count of cards held whose face is unknown |
| `table` | public | unknown faces marked `None` |
| `known_ranks` | public | unknown cards contribute nothing |
| `opponent_cards` | public | a count, never identities |
| `talon_size` | public | a count, never contents or order |
| `trump`, `attack_limit`, `phase`, `refill_to` | public | |

**Absent by construction:** opponent cards, talon contents, talon order,
hidden-card assignments, event history, any `GameState`. A reachability
audit walks every node of every tree and asserts that every reachable
`Card` has been observed by the searcher.

## Action classes

At an opponent node the branches are *classes of reply*, not moves:

| Class | Modelled effect |
|---|---|
| `BEAT` | one card leaves their hand and covers the open attack. Identity unknown, so the slot contributes **no new rank**. |
| `TAKE` | they pick up; the bout enters throwing-in and the searcher moves again. |
| `ADD` | they play an unknown card. The searcher would have to answer a card it cannot see → **frontier, branch stops**. |
| `END` | they end the bout → resolution. |

`OpponentAction` is a distinct type from `Move`, so an abstract action can
never be handed to the rules engine by accident.

### Why this is not inference

Offering *both* `BEAT` and `TAKE` is the opposite of inference: it assumes
nothing about what the opponent holds and keeps both possibilities. The
only thing consulted is their **hand size**, which is public — an opponent
with no cards cannot beat. Refusing to credit an unknown card with a rank
is likewise conservative: it can only understate the searcher's own future
options, never invent one.

## The tree is a strict conservative superset

Measured over 13,334 opponent nodes from real games:

- **real classes omitted: 0.** Every legal opponent move falls into a
  class the abstraction offers. It never blinds the search to a reply.
- **classes offered that were impossible: 3,311.** The abstraction
  routinely credits the opponent with replies they do not hold cards for.

So it is a **strict superset at the class level**, and additionally a
*coarsening*: many distinct beating cards collapse into one `BEAT` branch.

## The direction of the error, and what it buys

The opponent gets extra options; the searcher gets fewer (unknown ranks
are not credited). Both push the same way, so an abstract value is a
**lower bound** on the true value whenever no heuristic leaf contributes
to it. Hence an asymmetry that the API now names explicitly:

| Property | Sound? | Oracle check |
|---|---|---|
| `is_proven_win` | **Yes** — a real forced win | 358 of 358 claims confirmed |
| `is_lower_bound_loss` | **No** — not a proven loss | 60 of 221 claims were positions actually winnable or drawable |

Both were checked against a perfect-information oracle that reads the real
hidden state — something only a test may do. The naming exists so that a
later phase cannot mistake pessimism for proof.

**Correctness guarantee:** minimax and alpha-beta are *exactly correct
over the abstract tree* — verified by 10,876+ paired comparisons of score
and chosen move, plus hand-built trees whose values were computed by hand.
The abstract tree's values approximate the real game's, exactly in the
proven-win case and heuristically otherwise.

## Depth

`depth` counts **plies** — individual decisions by either side.

| `depth` | Meaning |
|---|---|
| 0 | evaluate the root; rejected at the root by `SearchConfig` |
| 1 | expand root moves, evaluate each child |
| 2 | shallowest depth at which a **minimising** node appears |
| n | n decisions from the root |

A branch stops early regardless of budget at a leaf: a resolved bout or an
unknown-card frontier. Those are properties of the position, not the
budget.

## Terminal scoring

Named constants, not magic numbers: `TERMINAL_WIN = +10000`,
`TERMINAL_LOSS = -10000`, `TERMINAL_DRAW = 0`, adjusted by ply so a sooner
win outscores a later one. Heuristic scores are bounded well inside
`MATE_MARGIN = 1000`, so a guess can never be mistaken for a proof.

**Which terminals are derivable at all?** Only at a resolved bout with an
**empty talon**. There, both players' card *counts* follow from public
information plus the searcher's own hand — no identity is inferred — so
win, loss and draw are real facts about counts. While the talon holds
cards, draws make the continuation unknown and the node is scored
heuristically.

## Evaluation

Every score is from the searcher's point of view at every node; the
minimax layer maximises at the searcher's nodes and minimises at the
opponent's. Evaluations are not side-to-move-relative, which removes a
whole category of sign errors.

The baseline is three terms: card advantage (dominant — you win by running
out), a small bonus per trump in the searcher's own hand, and a penalty
per attack it still must answer. The opponent's trumps are hidden and are
not guessed at. Unknown cards count as material but earn no trump bonus.

## Move ordering

`ai/ordering.py` orders by *promise*, so alpha-beta meets good moves first
and cuts off sooner. This is distinct from `ai/tiebreak.py`, which is an
arbitrary-but-fixed total order. Ordering changes node counts and never
results — tested by searching the same positions under both orderings.

Every comparison ends with the canonical key, so nothing falls through to
object identity, hashing or list order. No history heuristic, no killer
moves, no transposition table: those need search state carried between
nodes, and this phase is about getting the tree right first.

## SearchBot

Implements the Phase 4 `Agent` protocol unchanged. It builds a node from
the view, searches with the root restricted to exactly the moves it was
handed, and returns one of them. Restricting the root is what makes the
contract structural: **a move that was not offered cannot be returned**,
whatever the abstraction concludes. Fully deterministic; the RNG is
accepted and ignored.

## Known limitations

Stated plainly, because they matter for what comes next:

1. **A "proven loss" is not proven.** See the table above.
2. **The horizon is a bout.** Nothing past a resolution is searched,
   because draws are hidden. Multi-bout planning needs Phase 6+.
3. **Opponent replies are coarsened.** `BEAT` does not distinguish
   beating with a six from beating with the ace of trumps.
4. **Defender-side search is shallow.** When the opponent adds a card the
   branch stops immediately, so searching as defender sees little.
5. **The evaluation is crude** and deliberately so.
6. **Strength is not correctness — and here they point opposite ways.**

   SearchBot beats RandomBot 95%. It **loses to GreedyBot**, and it loses
   *worse the deeper it searches*:

   | depth | SearchBot vs GreedyBot (200 games, both seats) |
   |---|---|
   | 1 | 45.5% / 50.5% |
   | 3 | 32.5% / 36.0% |
   | 5 | 31.0% / 33.5% |

   Deeper search making play worse is the classic signature of optimising
   hard against a biased model, and it was predicted in
   [`ai-design.md`](ai-design.md) before any search existed. Two causes,
   both documented above: the abstraction is systematically pessimistic
   (the opponent is credited with replies they cannot make), and the
   evaluation is crude. Searching further just finds better ways to
   satisfy the wrong objective.

   This does not mean the search is incorrect. Minimax and alpha-beta
   agree exactly, hand-computed values match, and every proven win is
   real. It means the *model* is too impoverished to reward depth, which
   is a statement about what Phase 5 was allowed to know. The fix is not a
   deeper search or a tuned evaluation — it is hidden-state machinery, and
   that is Phase 6 onward.

   Reporting a bot that loses to its own baseline is the point of
   separating correctness from strength.

The honest summary: this is correct search over a deliberately impoverished
model of the game. Making the model less impoverished requires hidden-state
machinery that later phases will supply.


## Phase 7.3.2a: a second search, over one determinized world

### Why a second search exists

The search described above is an **information-set** search. Its
`SearchNode` holds the opponent as a *count* plus abstract action
classes, because the observer cannot see their cards. That was right for
what it had, and it is deliberately incapable of holding a hidden hand.

The consequence only became visible when determinized worlds arrived.
Routing a `DeterminizedPosition` through `observe()` to build a
`SearchNode` **re-redacts it**: the sampled world is thrown away and two
different worlds become observationally identical again. Measured
directly, two distinct sampled worlds produced a byte-identical Phase 5
result — same move, same score. Aggregating over worlds that way would
average the same number.

So Phase 7.3.2a searches a different space:

| | input | opponent | scope |
|---|---|---|---|
| Phase 5 | `InformationSet` → `SearchNode` | abstract, by action class | bout-local, conservative |
| Phase 7.3.2a | `DeterminizedPosition` | real cards, *within one hypothesis* | the whole hypothetical game |

Both are correct for their own space; neither replaces the other, and
`SearchBot` is untouched.

### What it may see

Every card in the hypothetical position, opponent hand included — because
that position is a *hypothesis* sampled from the observer's own
information, not the real game. The real hidden state is never an input.

### Value convention and depth

Values are always from the **root player's** point of view, at every
node, whoever is to move; their nodes maximise, the opponent's minimise.
The root player need not be the mover. `depth` counts plies, matching
Phase 5, and `evaluate_position()` is the depth-0 case. Terminals are
detected **before** the depth cutoff and come from the frozen Phase 2
engine.

### Evaluator reuse

`BaselineEvaluator` is reused unchanged. It scores a `SearchNode`, so
`DeterminizedPosition.evaluation_node()` builds one — in the
materialisation boundary, the only module in `ai/` allowed to read state
fields. That node is used for **leaf scoring only**; its Phase 5
abstraction machinery is never consulted, and `kind=DECISION` ensures
`SearchNode.outcome()` cannot pre-empt terminal detection.

### Proven results

Unlike Phase 5, a proven result here is exact **within the hypothesis**:
the search sees every card, so a win it proves is a real forced win *in
that world*. It says nothing about the real game, where that world may
not be the true one — hence `is_proven_win` here versus Phase 5's
deliberately cautious `is_lower_bound_loss`.

### One world only

**No world aggregation exists yet.** No sampling of several worlds, no
averaging, no voting, no expected values, no selection across worlds.
That is Phase 7.3.2b, and it needs this primitive to be trustworthy
first. No playing-strength improvement is claimed.

### A process note worth keeping

A mutation run killed from outside left the "never decrement depth"
mutation in the source, because a `finally` block does not run on
`SIGKILL`. Checking that the file still *parsed* confirmed nothing, and
every measurement taken afterwards was wrong — a 12-second suite appeared
to take 280. The mutation runner now snapshots to disk before mutating
and restores unconditionally at startup.

The same incident exposed a real test weakness: that mutation was only
"caught" by hanging, which is not a detection. `test_max_depth_reached_
never_exceeds_the_budget` now runs on tiny endgames where an
un-decremented search still terminates, turning the same fault into
`AssertionError: minimax reached ply 6 with a budget of 2`.


## Phase 7.3.2b: multi-world aggregation

> This is determinization-based multi-world aggregation. It is not a
> claim of optimal imperfect-information play and does not eliminate
> strategy fusion.

### Why the root player must be the mover

Aggregation compares one canonical set of root moves across worlds, so
that set has to mean the same thing in every world. It does — but only
from the seat that is moving. Measured over 2,719 positions from real
games:

| observer | worlds gave identical root move sets |
|---|---|
| to move | 2,719 of 2,719 |
| not to move | differed in 1,593 positions |

The opponent's options depend on cards the observer cannot see, so from
the non-moving seat there is no common action set. `aggregate_worlds`
therefore **requires** the root player to be the player to move and
raises otherwise. It does not union, intersect, or take the first world's
moves. The precondition is proven, not assumed, and the rejection path is
tested.

### The formula

```
V(m) = Σ weight(w) · V(w, m)  /  Σ weight(w)
```

`V(w, m)` is the Phase 7.3.2a value of playing `m` in world `w`, from a
fixed root-player perspective. Weights are **relative** and normalised by
their sum, so scaling them all changes nothing.

### Weight semantics

Default weight is 1 for every world, which *is* the Phase 7.2 belief
model: it is uniform over compatible allocations, so every compatible
world carries equal probability. **No second probability model was
introduced.** Explicit `WeightedWorld` weights exist for callers that
have a distribution; negative weights and a zero total are refused.

### Per-world values

Computed by restricting the Phase 7.3.2a search to a single root move,
not by searching the child. That distinction caused a real bug: searching
the child restarts the ply counter, and terminal scores carry a ply
adjustment, so child-rooted values were off by one ply and single-world
equivalence broke. Restricting the root reuses 7.3.2a's semantics exactly
and makes the equivalence hold by construction.

### Enumeration and sampling

Exact enumeration of the allocation space is provided and cross-checked
against the Phase 7.2 reference, but guarded: an opening position has
~475,000 allocations, so `enumerate_worlds` **raises** above a documented
limit rather than truncating silently. Sampling via `WorldGenerator` is
the general path; under the uniform model sampled worlds are equally
weighted.

### Tie-breaking and determinism

Ties go to whichever move comes first in the deterministic search
ordering — the same rule `search_world` uses. Two tie-break rules in one
codebase would make single-world equivalence hold only by coincidence,
which is exactly what happened before they were unified.

Worlds are summed in canonical key order, not caller order, so
permutation invariance is exact rather than resting on the data.
Floating-point addition is not associative; no instability was observed
across 2,440 permutation checks on real positions, which is precisely why
the ordering is asserted directly — the aggregate alone would not reveal
its loss. No cache exists, so cross-world contamination is structurally
impossible.

### Scaling

Linear in world count, as expected with no cache:

| depth | ms/world (1 world) | ms/world (25 worlds) | nodes/world |
|---|---|---|---|
| 1 | 0.68 | 0.52 | 9.3 |
| 3 | 1.47 | 1.48 | ~42 |
| 5 | 4.51 | 4.88 | ~174 |

### Limitations

Averaging perfect-information values over sampled worlds solves each
world as though the searcher would know which world it is in. It will
not. That is **strategy fusion**, recorded in
[`ai-design.md`](ai-design.md) since Phase 1 and not addressed here.
There is no opponent model, so worlds are weighted by the epistemic
uniform prior alone and nothing is inferred from how the opponent has
played. `SearchBot` does not consume this layer, and no playing-strength
claim is made or measured.
