# DurakFish — Architecture

## 1. The one rule

Dependencies point in exactly one direction. Nothing lower may import
anything higher.

```
cards/          suits, ranks, cards, decks          (no dependencies at all)
  ↓
game/           rules, moves, state, engine, history (depends on cards)
  ↓
information/    card tracker, information set, opponent model
  ↓
ai/             evaluation, move ordering, minimax, MCTS, ISMCTS, endgame solver
  ↓
interface/      CLI, GUI, analysis output
vision/         screen observation — plugs into information/, not into game/
```

`vision/` sits off to the side on purpose. It produces *observations*
(`"a 9♠ appeared on the table"`) and hands them to the card tracker. It
never touches rules, never decides legality, never talks to search. If
the vision module is deleted, everything else still runs.

Enforcement is by review and by `mypy`/import discipline, not by
convention alone. A single upward import (rules importing an evaluator,
say) is what turns an engine into a script.

## 2. Layer responsibilities

| Layer | Owns | Explicitly does **not** own |
|---|---|---|
| `cards` | card identity, notation, the talon | trump, beating, legality |
| `game` | legality, transitions, history, terminal conditions | heuristics, probabilities, search |
| `information` | what a player may know; belief over hidden cards | choosing moves |
| `ai` | evaluation, search, move choice | ground truth about hidden cards |
| `interface` | input/output | any rule of the game |

The rules engine contains **no AI logic**, and the AI reaches the rules
only through `get_legal_moves(state, player)` / `apply_move(state, move)`.
That is what makes it possible to swap in a 52-card or 4-player variant
without touching search.

## 3. Core invariants

These hold at every point in every phase, and the tests exist to defend
them:

1. **Conservation.** Every card is in exactly one place: a hand, the
   table, the discard pile, or the talon. Never two, never zero.
2. **Determinism.** `(deck order, seed, sequence of decisions)` fully
   determines a game. No component reads the global `random` module.
3. **No leakage.** In non-omniscient modes, an agent's inputs are its
   `InformationSet` and nothing else. Enforced structurally: the agent
   API takes an information set, not a `GameState`, so cheating is a
   type error rather than a matter of self-discipline.
4. **Immutable transitions.** `apply_move` returns a new state; the
   input state is untouched. Search relies on this.

Invariant 3 is worth restating: the honest way to prevent cheating is not
to *ask* the AI to ignore the opponent's hand, but to make it impossible
to see it. That is why `InformationSet` is a first-class type and why
`GameState` never reaches an agent in real play.

## 4. Project tree

Implemented in Phase 1 is marked ✅; the rest is the planned shape.

```
durakfish/
├── pyproject.toml                    ✅
├── README.md  LICENSE  requirements.txt  .gitignore   ✅
├── docs/
│   ├── architecture.md               ✅  (this file)
│   ├── agents.md                     ✅  the baseline agent contract
│   ├── search.md                     ✅  search, and the abstraction's limits
│   ├── knowledge.md                  ✅  observed / deduced / probable
│   ├── rules.md                      ✅  the rules, and every ambiguity
│   └── ai-design.md                  ✅  algorithm analysis
├── src/durakfish/
│   ├── __init__.py                   ✅
│   ├── py.typed                      ✅
│   ├── exceptions.py                 ✅
│   ├── config.py                         config loading (Phase 3)
│   ├── cards/
│   │   ├── enums.py                  ✅  Rank, Suit
│   │   ├── card.py                   ✅  Card, parsing, formatting
│   │   └── deck.py                   ✅  Deck, seeded shuffling
│   ├── game/
│   │   ├── moves.py                  ✅  Attack/Defend/Take/EndAttack
│   │   ├── ruleset.py                ✅  variant configuration
│   │   ├── rules.py                  ✅  beats(), legality, bout logic, new_game
│   │   ├── state.py                  ✅  GameState, phases, validate(), keys
│   │   └── history.py                ✅  persistent event log
│   ├── information/
│   │   ├── information_set.py        ✅  redaction: what a player may know
│   │   ├── knowledge.py              ✅  locations, certainty, KnowledgeState
│   │   ├── deduction.py              ✅  five justified logical rules
│   │   ├── tracker.py                ✅  CardTracker, incremental
│   │   ├── belief.py                 ✅  exact uniform-allocation model
│   │   ├── search_contract.py        ✅  SearchInformation (Phase 7.1)
│   │   ├── worlds.py                 ✅  determinized worlds (Phase 7.2)
│   │   └── opponent_model.py             behavioural modelling (later)
│   ├── ai/
│   │   ├── base.py                   ✅  Agent protocol (info set → move)
│   │   ├── searchnode.py             ✅  conservative search abstraction
│   │   ├── evaluation.py             ✅  terminal + baseline scoring
│   │   ├── ordering.py               ✅  search move ordering
│   │   ├── search.py                 ✅  minimax / alpha-beta
│   │   ├── search_bot.py             ✅  SearchBot
│   │   ├── determinized.py           ✅  materialisation boundary (7.3.1)
│   │   ├── tiebreak.py               ✅  canonical move order
│   │   ├── random_bot.py             ✅  uniform baseline
│   │   ├── greedy_bot.py             ✅  deterministic baseline
│   │   ├── probability_bot.py            (Phase 7)
│   │   ├── evaluator.py                  configurable evaluation
│   │   ├── ordering.py                   move ordering
│   │   ├── minimax.py                    alpha-beta + transposition table
│   │   ├── mcts.py  ismcts.py            simulation search
│   │   ├── endgame.py                    exact solver, empty talon
│   │   └── durakfish.py                  the combined engine
│   ├── simulation/
│   │   ├── record.py                 ✅  GameRecord, replayable
│   │   ├── runner.py                 ✅  play_game / replay / verify
│   │   ├── stats.py                  ✅  minimal match summary
│   │   └── self_play.py  tournament.py  elo.py       (Phases 12-13)
│   ├── vision/
│   │   ├── capture.py  recognizer.py  observer.py  input_controller.py
│   ├── interface/
│   │   ├── cli.py  gui.py  analysis.py  replay.py
│   └── utils/
│       ├── serialization.py  logging.py  stats.py
└── tests/
    ├── test_cards.py                 ✅
    ├── test_deck.py                  ✅
    ├── test_properties.py            ✅
    ├── test_moves.py  test_rules_beats.py  test_legal_moves.py   ✅
    ├── test_bout.py  test_state.py  test_game_properties.py      ✅
    ├── test_adversarial.py           ✅  independent audit
    ├── test_information_set.py       ✅  leakage, indistinguishability
    ├── test_agents.py  test_runner.py                ✅
    ├── test_architecture.py          ✅  the layer rule, executable
    ├── test_bots.py                  ✅  RandomBot / GreedyBot behaviour
    ├── test_phase4_audit.py          ✅  boundary + determinism for real bots
    ├── test_phase2_frozen.py         ✅  SHA-256 freeze
    └── test_tracker.py  test_probability.py  test_minimax.py  test_mcts.py
```

Two departures from the tree in the original spec, both deliberate:

- **`ai/ordering.py` and `ai/endgame.py` are separate modules.** Move
  ordering is used by minimax *and* by MCTS priors, so it does not belong
  inside `minimax.py`. The endgame solver is exact and structurally
  different from the approximate searchers (see `ai-design.md` §5).
- **`simulation/dataset.py` exists from the start.** The self-play runner
  should write training records as a side effect from day one; retrofitting
  logging into a runner after 100k games have been thrown away is wasted work.

## 5. Phase 1 design decisions

**Rank covers 2–A, not just 6–A.** The 36-card deck is the *deck's*
business, not the vocabulary's. `standard_cards(36)` filters to 6–A;
`standard_cards(52)` takes everything. Had `Rank` started at six, adding
a 52-card variant later would renumber every card code and invalidate
every serialized game. Sizes 20, 24, 36 and 52 all work today.

**Card code is a dense integer 0–51, and `hash(card) == card.code`.**
Cards are dictionary keys in probability tables and components of
transposition hashes, so hashing must be cheap and collision-free. The
code doubles as an array index for future bitboard work.

**Cards are interned.** `Card.of`, `Card.from_code` and `Card.parse`
return shared instances from a 52-element table. Equality is still by
value, so a hand-built `Card(Rank.ACE, Suit.SPADES)` compares equal to
the interned one — interning is an optimisation, never a correctness
requirement.

**`Card` has no `beats()` and no `<`.** Beating depends on the trump,
which is game state; putting it on `Card` would invert the dependency
direction. And a silent `card_a < card_b` that looks like "is weaker" but
ignores trump is the kind of bug that produces a plausible-looking engine
that plays badly. Comparison is deliberately a `TypeError` (there is a
test for it). `sort_cards()` exists for display only.

**Suit order is canonical, not strength.** Alphabetical, for deterministic
serialization and stable display.

**The deck knows about a bottom card, not about a trump.** `bottom_card`
is a neutral fact about a pile; the rules layer decides it means trump.
Real Durak turns the trump up after dealing — shuffling once and dealing
from the top is distributionally identical and has no special cases.

**Shuffling requires an explicit `random.Random`.** There is no path to
the global RNG anywhere in the engine, so a seed reproduces a game
exactly. `Deck.shuffled()` with no seed is allowed and documented as
non-reproducible.

**`reset()` restores the last snapshot, and `shuffle()` refreshes that
snapshot.** So a shuffled, dealt deck resets to the *shuffled* order.
This ambiguity was found by a failing test, not by inspection — the
original implementation reset to new-deck order, which is almost never
what a caller wants.

## 6. What Phase 1 deliberately leaves out

No logging setup (arrives with the engine in Phase 3), no config loading
(Phase 3), no bitmask hand representation (Phase 14, after profiling —
optimising before there is a benchmark is guesswork), no localisation of
card notation. Cyrillic input tokens are specifically excluded: Cyrillic
`К` and Latin `K` are identical-looking homoglyphs, so accepting both
would make parse bugs invisible in a terminal. Localisation belongs in
the interface layer where it can be tested against real input.


## 7. Phase 3: the information boundary

Phase 3 turned the rules engine into a driveable simulator and, more
importantly, installed the boundary every later phase plugs into:

```
GameState (ground truth, never leaves the driver)
    │  observe(state, player)      ← redaction
    ▼
InformationSet ──► Agent.choose(view, legal) ──► Move
    │                                              │
    └──────────── driver applies ◄─────────────────┘
                        │
                        ▼
                   GameRecord ──► replay / verify
```

**Why the boundary exists before the first bot.** A driver must hand an
agent *something*. Had that been a `GameState`, every bot from Phase 4
onward would have been built on a cheating-capable interface, and Phase 6
would have been a rewrite of the agent protocol plus every agent using
it. Introducing `InformationSet` here costs a projection function; adding
it later would have cost a migration.

**Redaction is not deduction.** `observe()` reports what a player has
seen. It never works out where a card *is*. A card the opponent took is
recorded as observed — you watched it — but nothing claims they still
hold it. `CardTracker` in Phase 6 consumes this and does the deducing.
The split is enforced by tests, not by intention.

**The driver sits above the AI layer**, in `simulation/`, not in the
`game/engine.py` the original tree proposed. A driver that knows about
agents cannot live inside the rules package without inverting the
dependency arrow. `game/engine.py` was dropped rather than left as a
misleading placeholder.

**The layer rule is executable.** `tests/test_architecture.py` parses the
import graph and fails on any upward import, and separately checks that
importing `durakfish.game` does not pull in `information`, `ai` or
`simulation`. A comment in a design document enforces nothing.

**Determinism.** One master seed fans out into a deal stream and one
stream per agent, all recorded. Separate streams mean an agent's
randomness cannot perturb the deal, so two bots can later be compared on
identical cards. The driver itself draws no random numbers.

### Known cost

`observe()` with history is linear in the event log: ~51 us early in a
game, ~349 us late, against ~14 us for `apply_move`. It therefore
dominates a long game, and the driver runs at ~40 games/sec with history
versus ~134 without. This is acceptable at one call per turn and would
not be at search-node rates. The fix — maintaining the redacted log
incrementally — belongs to `CardTracker` in Phase 6, where the same walk
is needed anyway. Until then, `play_game(observe_history=False)` is
available for agents that ignore history.


## 8. Phase 4: the baseline agents

Phase 4 put real policies behind the Phase 3 boundary. The pipeline the
phase existed to demonstrate now runs end to end for actual agents:

```
GameState → observe() → InformationSet + legal moves → Agent → Move → GameState
```

`RandomBot` and `GreedyBot` are documented in
[`agents.md`](agents.md). Two architectural points are worth recording
here.

**The bots are the cheapest part of a decision.** `choose()` costs 1.8-2.8
us against `observe()`'s 186 us. The information boundary is still the
bottleneck, unchanged from Phase 3, and the fix is still Phase 6's
incremental tracker. Nothing in the agent layer needs optimising, and
saying so is more useful than tuning it.

**Canonical move order is not search move ordering.** `ai/tiebreak.py`
provides a total order over moves derived only from their contents. It
exists so that tie-breaking cannot depend on object identity or list
order, and so that "uniform over the legal moves" is a statement about
the move *set*. Ranking moves by how promising they are, to help
alpha-beta cut off sooner, is a different job for a later phase and gets
its own module.

The Phase 3 public interfaces were not changed. `game/` was not touched;
its SHA-256 freeze still passes.


## 9. Phase 5: search over a conservative abstraction

Phase 5 added minimax and alpha-beta behind the same information
boundary. The full design and its limits are in [`search.md`](search.md);
three points belong here.

**The tree is not the game tree.** A full-width tree cannot be built from
an information set — opponent replies need their cards, and anything past
a bout resolution needs the talon order. Determinization was out of
scope, so the search models the opponent by *action class* and stops at
frontiers it cannot see past. The result is a strict conservative
superset: measured over 13,334 opponent nodes, it omitted 0 real options
and offered 3,311 impossible ones.

**The error has a direction, and the API names it.** Extra opponent
options and fewer own options both push the same way, so abstract values
are a lower bound. `is_proven_win` is sound (358/358 confirmed against a
perfect-information oracle); `is_lower_bound_loss` is not (60 of 221
claims were positions actually winnable). Naming them apart stops a later
phase mistaking pessimism for proof.

**Deeper search plays worse.** SearchBot loses to GreedyBot, and by more
at depth 5 than depth 1. That is model bias being optimised harder, not a
search bug, and it is the clearest possible argument for why the hidden-
state work in the phases ahead is the real prerequisite for strength.


## 10. Phase 6: the knowledge layer

Phase 5 ended with a search that was correct and weak, because its model
of the opponent was too pessimistic and nothing could tell the difference.
Phase 6 builds the missing piece. Full detail is in
[`knowledge.md`](knowledge.md); three architectural points belong here.

**Three categories, structurally separated.** Observed, deduced and
probable each have their own representation, and the pipeline runs one
way: `observe() -> CardTracker -> deduce() -> KnowledgeState ->
BeliefState`. Beliefs are computed *from* knowledge and cannot write back
into it, so a likelihood can never harden into a fact. There is a test
that finds cards with marginals above 0.8 and asserts they still report as
`UNKNOWN`.

**`observe()` was not touched.** It remains a pure projection performing
no inference. Knowledge consumes an information set rather than rewriting
one, so the Phase 3 boundary is exactly where it was — which is why all
60 Phase 3 audit tests still pass unchanged. The architecture test now
also asserts that the four modules *downstream* of redaction never name a
`GameState`, while `information_set.py` is exempt by design, since
`observe()` is the boundary and necessarily takes ground truth.

**Deduction and belief are separate modules on purpose.** Every deduction
rule is named and individually justified from the rules of Durak; anything
merely plausible is refused entry and left to the belief model. The
temptation to let one function do both is exactly how a heuristic ends up
being reported as a certainty.

Nothing consumes this layer yet. No agent was changed, and that was
deliberate: mixing "is the information correct?" with "does it help?"
would have made both questions harder to answer.


## 11. Phase 7.1: the search contract

One immutable object, `SearchInformation`, composing the observed view,
the knowledge and the beliefs. It is the single door into a future
belief-aware search. Detail in [`knowledge.md`](knowledge.md).

Two decisions are worth recording here.

**It lives in `information/`, not `ai/`.** It contains no AI logic and is
assembled entirely from information-layer objects, so putting it here
keeps `ai/` from having to construct its own inputs — which is exactly the
situation in which somebody eventually reaches for a `GameState`.

**It adds no new information model.** Everything in it was built and
audited in Phase 6. A separate `DeductionState` was considered and
rejected: `KnowledgeState` already carries the justifying rule on every
placement, and a parallel type would only give the two copies a chance to
diverge.

`SearchBot` is untouched and its decisions are byte-identical, verified
over 1,500+ positions and whole recorded games.


## 12. Phase 7.2: determinized worlds

A `WorldGenerator` turns a `SearchInformation` into concrete hidden
worlds drawn uniformly from those the information permits. Detail in
[`knowledge.md`](knowledge.md); three points belong here.

**It lives in `information/`, and takes only a `SearchInformation`.** The
generator has no route to a `GameState`, which the noninterference test
proves by generating from two different *real* hidden states and
requiring byte-identical output.

**The validator is written independently of the sampler**, so it is
capable of failing it. Mutation testing found that two of its seven
checks are logically implied by the others and can never fire first; both
are documented as defensive in `worlds.py` rather than quietly dropped or
claimed as tested.

**A world is not a game state.** `DeterminizedWorld` holds two card
collections and nothing else. Materialising a position that search can
step through requires a `GameState` constructor, which does not belong in
the information layer — so that is Phase 7.3's problem, above this line.


## 13. Phase 7.3.1: the materialisation boundary

`ai/determinized.py` turns a `SearchInformation` plus one
`DeterminizedWorld` into a complete hypothetical position that the frozen
rules engine can search. Detail in [`knowledge.md`](knowledge.md).

The architectural point: the codebase now has **two boundaries, and only
two**, where information and ground truth meet.

```
GameState  --observe()-->  InformationSet  --build_position()-->  hypothesis
           information/information_set.py    ai/determinized.py
             the REDACTION boundary        the MATERIALISATION boundary
```

Each is the sole module in its package allowed to name a `GameState`, and
every module between them is banned from doing so. That symmetry is now
enforced rather than described: `test_phase3_audit.py` exempts exactly
one file in `ai/`, and a companion test fails both if the exempt list
grows and if the exemption stops being necessary.

The adapter reuses the Phase 2 engine wholesale — no second rules
implementation exists — and a position built from the *true* world has
the same `transposition_key` as the real state it came from, so
everything computed downstream is identical by construction.
