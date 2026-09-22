# Knowledge

Phase 6 builds the layer that Phase 5 showed was missing: a rigorous
account of what a player knows, what follows from it, and what is merely
likely.

## Why this phase exists

Phase 5's search was correct over its abstraction and still played badly —
losing to GreedyBot, and worse the deeper it looked. The cause was that
the abstraction credited the opponent with replies they often could not
make, and nothing in the system could tell the difference. Fixing that
needs knowledge, not a better search.

**Phase 6 is not about making SearchBot stronger.** No agent's strategy
was changed. The goal is an information model that is correct, so that
later phases have something sound to build on.

## Three categories, never conflated

| | Meaning | Example |
|---|---|---|
| **Observed** | the player saw it | own hand, the table, the discard pile, the face-up trump |
| **Deduced** | proven from public facts and the rules | a card the opponent took and has not played is still in their hand |
| **Probable** | possible, not proven | this card is in their hand with probability 0.4 |

The pipeline enforces the separation structurally:

```
observe()  →  CardTracker  →  deduce()  →  KnowledgeState  →  BeliefState
 (Phase 3)     accumulate      logic        facts only        probabilities
```

Each stage is a pure function of the one before it, and **beliefs cannot
write back into knowledge**. A 0.95 marginal is reported as
`Certainty.UNKNOWN`, not as a fact — there is a test that finds
high-probability cards and asserts exactly that.

`observe()` was **not modified**. It is still a pure projection performing
no inference, so the Phase 3 boundary is precisely where it was. Knowledge
is a layer that *consumes* an information set, never one that rewrites it.

## Certainty levels

Two questions, two methods:

- `certainty(card)` → `KNOWN` / `DEDUCED` / `UNKNOWN` — how we know where
  a card is.
- `location_certainty(card, location)` → `KNOWN` / `DEDUCED` / `POSSIBLE`
  / `IMPOSSIBLE` — what we know about one placement.

So "probability zero" (`IMPOSSIBLE`) and "not established"
(`UNKNOWN` / `POSSIBLE`) are distinguishable, as required.

## Locations

`SELF_HAND`, `OPPONENT_HAND`, `TALON`, `TABLE`, `DISCARD`. Only the middle
two are hidden, and they are kept **distinct**: an unobserved card is not
simply "an opponent card", and treating it as one is the mistake this
layer exists to prevent.

## The deduction rules

Five, each named, each justified from the rules alone. Every settled card
records which rule placed it.

| Rule | Justification |
|---|---|
| `observed` | the player saw it |
| `trump-at-talon-bottom` | the face-up trump is the bottom card and is drawn last, so while the talon is non-empty it is in the talon |
| `opponent-retains-taken` | a card seen taken and not seen played is still in their hand — cards leave a hand only by being played publicly |
| `talon-exhausted` | no talon slots remain, so every undetermined card is in their hand |
| `opponent-hand-saturated` | their hand is fully accounted for, so every undetermined card is in the talon |

The last two are the same counting principle in opposite directions.

Both non-trivial rules were validated against ground truth over 44,076
observations before being written, and every settled placement is checked
against ground truth over **100,000+ assertions** in the test suite.

### What is deliberately not deduced

Nothing is inferred from the opponent's *choices*. Declining to beat a
card suggests they lack a beater; it does not prove it, because taking is
always legal and sometimes correct. That is a belief, not a deduction, and
this engine does not make it. Discarding licenses no inference beyond the
card's own location.

## The belief model, stated exactly

After deduction, `n` cards remain undetermined, `k` of them in the
opponent's hand. The observer has no card-specific information
distinguishing them — anything that did would have been a deduction — so:

> every allocation of the undetermined cards to the two hidden locations
> is equally likely.

Uniform over `C(n, k)` allocations. **Exact, not an approximation.**
Marginals are `k/n`. Validated empirically against ground truth over
~1.4 million card-observations; predicted matched actual across the full
range 0.0 to 1.0.

### Cards are not independent

The dependence is real and the model respects it. With 4 undetermined
cards and 2 opponent slots, each is theirs with probability 1/2, but two
specific cards are *both* theirs with probability

```
(2/4) × (1/3) = 1/6      not     (1/2)² = 1/4
```

`probability_all_in` computes this from the hypergeometric and never by
multiplying marginals. Checked against a brute-force enumeration reference
in `tests/reference_belief.py`, which is kept out of the shipped library
precisely so nobody reaches for it on a hot path.

## Incrementality

Almost everything a deduction needs can be re-read from the current
position. Exactly one fact cannot, and it is the only thing carried
forward: **which cards the observer watched the opponent take and has not
seen played since.**

Because that set is the whole of the carried state, incremental tracking
and full rebuilding are provably equivalent — and the suite checks that
after *every event* of thousands of games rather than assuming it.

## Performance

| | |
|---|---|
| incremental update (per event) | 37.5 µs |
| full rebuild (per position) | 57.5 µs |
| `deduce()` | 28.5 µs |
| `probability()` per card | 0.64 µs |
| `key()` | 9.1 µs |
| serialise / deserialise | 28.7 / 64.9 µs |
| replay a whole game | 3.85 ms |

Incremental tracking is about **1.5× cheaper** than rebuilding — a real
but modest saving, because most of the cost is the deduction pass, not the
history walk. For comparison, `observe()` alone costs 186 µs, so knowledge
maintenance is roughly a fifth of what redaction already costs and is not
the bottleneck.

## Phase 7.1: the information-aware search contract

### Why it exists

Phase 5's search reasoned from raw observation and played badly, because
it credited the opponent with replies they often could not make. Phase 6
built the knowledge that would let it know better. `SearchInformation` is
the connector — the single object a belief-aware search will be handed.

Bundling matters more than it looks. Without one door, each future search
assembles its own inputs, and the day one of them reaches for a
`GameState` "just to get the talon size", nothing catches it.

### What it contains

Nothing new. It composes three things Phase 6 already audited:

| | |
|---|---|
| `view` | what was **observed** (Phase 3 redaction) |
| `knowledge` | what is **certain** or **deduced**, with the rule for each |
| `belief` | what is **probable**, and only that |

There is deliberately **no separate `DeductionState`**: Phase 6 records
the justifying rule on every placement inside `KnowledgeState`, so a
parallel type would duplicate it and invite the copies to disagree.

### What it cannot contain

No `GameState`, no opponent hand, no talon contents, no hidden history,
no callback that could fetch any of them.
`FORBIDDEN_ATTRIBUTES` names the escape hatches as data, so the rule is
machine-checkable; a test walks the entire reachable object graph and
asserts none exist.

This is also why `KnowledgeState.hidden_cards` was renamed to
`hidden_card_count` in this phase. It returns an integer, but the old
name read like a way of getting at hidden cards and collided with the
forbidden list. Renaming it kept the ban meaningful and made it match
`InformationSet.hidden_card_count`. It is the only Phase 3–6 public name
changed.

### Why SearchBot must not receive GameState after observation

`observe()` is the redaction boundary. Anything downstream that takes
ground truth makes the boundary decorative: the code would work, play
well, and be solving a game nobody will ever play against it. The
architecture test permits `GameState` in `information_set.py` — where
`observe()` necessarily takes it — and nowhere else downstream.

### The four levels

`classify(card)` returns exactly one of:

| | |
|---|---|
| `CERTAIN` | directly observed. Treat as fact. |
| `DEDUCED` | proven from public facts and the rules. Also fact. |
| `PROBABILISTIC` | not established; ask `probability()` |
| `UNKNOWN` | not established and no distribution applies |

A card in the opponent's hand with probability 0.99 classifies as
`PROBABILISTIC`, never `CERTAIN`. A test finds such cards and asserts it.

### Hidden-state noninterference

If two hidden worlds look identical to an observer, the contracts built
from them are identical — checked not by object equality but by a
13-channel fingerprint covering serialisation, keys, possible sets,
classifications, justifications, marginals, allocations, entropy and
`repr`, across opponent-hand, talon, combined and hidden-history
permutations.

### Cost

| | |
|---|---|
| `from_tracker()` (tracker current) | 1.3 µs |
| `from_view()` (rebuilds the tracker) | 71.9 µs |
| `fingerprint()` | 147 µs |
| `classify()` per card | 0.35 µs |

**`from_tracker` is ~56× cheaper**, which is direct guidance for Phase
7.2: maintain one tracker per observer and snapshot it, rather than
rebuilding per node. `fingerprint()` is an audit tool, not a hot path.

### Why determinization is deferred

Phase 7.1 defines what may cross the boundary; Phase 7.2 decides what to
do with it. Building a sampler at the same time as its information source
would mean debugging both together, and any disagreement between them
would be ambiguous. `allocations` — the number of hidden worlds still
consistent with what is known — is exposed here because a sampler will
need it. Exposing the count samples nothing.

## Phase 7.2: determinized worlds

### What a determinized world is

Committing to one concrete possibility for everything the observer cannot
see: which cards the opponent holds, and which cards are in the talon in
what order. A `DeterminizedWorld` is pure information-level data — two
card collections — and holds no `GameState`.

### Why generate them

Perfect-information machinery needs a fully specified position. Sampling
worlds is how an imperfect-information game is reduced to one. Phase 7.2
proves the worlds are correct; nothing consumes them yet.

### What makes a world legal

A world must be a member of `W(I)`, the set of hidden states compatible
with the observer's information. Seven conditions, all enforced by an
independently written validator:

1. **Conservation** — every card in exactly one place; no duplicates,
   missing or phantom cards.
2. **Sizes** — opponent hand and talon exactly as large as public
   information says.
3. **Certain ownership** — Phase 6's certain and deduced placements hold.
4. **Impossible placements** — never assigns a card where Phase 6 rules
   it out.
5. **Public preservation** — the observer's hand, table and discard are
   untouched.
6. **Trump position** — the trump card is the *bottom* card of a
   non-empty talon.
7. **Talon integrity** — no card appears twice in the ordered talon.

Two of these are **defensive rather than load-bearing**, established by
mutation testing and worth stating plainly: the *talon size* check is
implied by conservation plus the hand-size check, and the *impossible
placement* check is implied by public-trespass, overlap and
certain-ownership, so neither can ever be the first to reject a world.
They stay because those implications hold only while the surrounding
checks do.

### Trump-bottom ordering

`GameState.talon` is ordered and the trump card is always its bottom
card — confirmed across 22,646 non-empty talons with zero violations. But
`GameState.validate()` does not check it, so a sampler ignoring it would
produce worlds the engine accepts and that could never arise in play. The
world validator enforces it explicitly, and `game/` was not modified to
add the check, since Phase 2 is frozen.

### Why joint allocation, and why marginals are wrong

Phase 6 can say an undetermined card is the opponent's with probability
`k/n`. Sampling each card independently from that marginal is wrong,
because the cards are **dependent**: with 4 undetermined cards and 2
opponent slots each is theirs with probability 1/2, but two specific
cards are both theirs with probability **1/6, not 1/4**. Independent
sampling would also produce wrong-sized hands constantly, since nothing
holds the total to `k`.

The generator draws a uniformly random `k`-subset — a member of the same
combinatorial space Phase 6's probabilities are defined over. A test
measures the pair frequency at 20,000 draws and requires it to match 1/6
and to differ from 1/4 by a wide margin, so a regression to marginal
sampling fails loudly.

### Reusing Phase 6, not duplicating it

No second probability model was written. The generator reads
`candidates`, `opponent_slots`, `talon_slots` and `allocations` from the
Phase 6 knowledge that Phase 7.1 already bundles.

### Uniformity, and why no behavioural inference

Where nothing distinguishes two compatible allocations, they get equal
probability. No card-strength weighting, no strategic weighting, no
opponent model. An opponent declining to beat a card is evidence they
lack a beater, but taking is always legal and sometimes correct, so
acting on it means choosing a model of the opponent — which this phase
deliberately does not do. Weighted sampling needs a correct uniform
sampler underneath it in any case.

### Seeded reproducibility

Sampling takes an explicit `random.Random`. Nothing touches the global
module, so a seed reproduces a world sequence exactly across generator
instances and processes.

### Cost

| position | sample one world | validate | reference enumeration |
|---|---|---|---|
| early (29 undetermined, 475,020 allocations) | 25.7 µs | 44.9 µs | infeasible |
| late (8 undetermined, 56 allocations) | 8.8 µs | 17.7 µs | 1.44 ms |
| determined (1 allocation) | 2.1 µs | 8.4 µs | 16 µs |

Marginal cost is flat — sampling 100 worlds costs the same per world as
sampling one — so generator setup is not a hidden overhead. Building a
generator and 32 worlds costs about 821 µs.

### Why search integration is deferred

Phase 7.2's only claim is that the generated worlds are correct. Wiring
them into search means choosing how many worlds, how to aggregate across
them, and how to avoid strategy fusion — decisions that need a
trustworthy sampler underneath. **No playing-strength improvement is
claimed or measured here**, and `SearchBot` is untouched.

### What Phase 7.3 will need to solve

Turning a world into a position the existing search can step through
requires materialising a `GameState`, which the information layer must
not do — so that construction belongs above it. Beyond that: how many
worlds to sample, how to combine per-world results without assuming the
searcher will know which world it is in, and the strategy-fusion problem
recorded in `ai-design.md` back in Phase 1.

## Phase 7.3.1: evaluating one determinized world

### Real state versus hypothetical state

Two things are now both "a complete position", and confusing them would
undo every guarantee since Phase 3.

A **real `GameState`** is ground truth — what actually happened. Nothing
downstream of `observe()` may read it.

A **`DeterminizedPosition`** is also complete, and does contain cards the
observer cannot see. But every one came from a `DeterminizedWorld`
sampled from the observer's own information. It is a *hypothesis*: one of
the worlds consistent with what the observer knows. Knowing all the cards
inside a hypothesis is the entire point; knowing them in reality is
cheating.

### Why determinized worlds are legal search inputs

Phase 7.2 proved every generated world is a member of `W(I)` — compatible
with everything the observer knows. A position built from one therefore
describes a game that *could* have happened, which is exactly what
perfect-information machinery needs.

### How the position is built

`build_position(SearchInformation, DeterminizedWorld)` takes the visible
position from the first and the hidden allocation from the second. Every
field comes from exactly one of the two inputs.

The result wraps a genuine `GameState`, so legality, transitions and
terminal detection are **the frozen Phase 2 engine, unchanged**. No
second game engine, no restated rules, nothing to keep in sync.

One legitimate difference: an observer's history has draws redacted, so a
hypothesis cannot reconstruct the real event log and carries none.
History is excluded from `transposition_key` and from legality, so
nothing search-relevant depends on it.

### Why `ai/determinized.py` may name `GameState`

It is the **materialisation boundary** — the mirror image of
`information/information_set.py`, the *redaction* boundary. One converts
ground truth into information; one converts information back into a
hypothesis. Each is the only module in its package permitted to name a
game state, and everything between them is banned from both. The
exemption is enforced narrowly: a test asserts the exempt list has not
grown *and* that the exemption is still needed.

### How equivalence is verified

When the world sampled happens to be the true one, the hypothetical
position must behave identically to the real one. That gives a ground
truth oracle, and equivalence is checked at four levels — agreeing on one
proves little about the others:

| level | result |
|---|---|
| position identity (`transposition_key`) | 2,000+ positions, exact |
| legal action sets | 2,000+ positions, all four move types |
| transitions, recursively three plies | 3,000+ comparisons |
| terminal outcomes and full-depth perfect-information search | 100+ endgames |

Plus the shipped Phase 5 search at depths 1, 3 and 5, run on both
positions and compared on move, score **and node count**.

The test harness reads the real `GameState` to construct these
comparisons. Production code in `ai/determinized.py` never does, and a
static check enforces the difference.

### Noninterference and substitution

Two real games with identical observer information, given the same seed,
produce identical positions. Stronger: supply the *same world object* to
both, and evaluation is identical — `SI(G1) == SI(G2)` and `W1 == W2`
implies `Eval(SI(G1), W1) == Eval(SI(G2), W2)`. That rests on nothing
about the sampler.

### Invalid worlds are refused

The Phase 7.2 validator runs by default. Dropped cards, duplicates, wrong
hand or talon size, impossible ownership and trump-not-at-bottom all
raise rather than being repaired. Validation can be disabled explicitly
for hot paths; it is never off by accident.

### Cost

| | |
|---|---|
| `build_position()` validating | 56.5 µs |
| `build_position()` validation off | 26.2 µs |
| `legal_moves()` | 2.2 µs |
| `apply()` | 12.4 µs |

Adapter overhead against searching the real position directly is a flat
**~57–61 µs per position**, roughly half of it validation. As a share it
falls as depth grows — +85% at depth 1, +29% at depth 5 — because it is a
per-position constant while search cost is not.

### Why multi-world aggregation is deferred

Phase 7.3.1 evaluates **one world at a time**, and makes no decision from
several. Combining per-world results without assuming the searcher will
know which world it is in is the strategy-fusion problem recorded in
[`ai-design.md`](ai-design.md) since Phase 1, and it deserves to be
attacked on top of a contract already known to be correct. **No
information-aware playing strength is claimed**, and `SearchBot` is
untouched.

## Known limitations

1. **No behavioural inference.** The strongest available signal about an
   opponent's hand — how they play — is deliberately unused. Adding it
   means choosing an opponent model, and this phase makes none.
2. **Marginals are uninformative among candidates.** Every undetermined
   card shares the same marginal, by construction. `most_likely_in` says
   so rather than implying a ranking exists.
3. **The model is exact only for the stated assumption** — uniform
   shuffling and conditioning on public information. It is exact for the
   game as this engine deals it.
4. **Nothing consumes this yet.** No agent was changed. Wiring beliefs
   into search is the next phase's work, and doing it here would have
   mixed up "is the information correct?" with "does it help?".
