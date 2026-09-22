# DurakFish

An engine for **Podkidnoy Durak** (Подкидной дурак), built the way a
serious game engine is built: rules first, then information tracking,
then probability, then search.

Durak is an imperfect-information game, so the interesting engineering is
not the search but the question *what does the engine legitimately know?*
DurakFish answers that structurally — an agent receives an
`InformationSet`, never a `GameState`, so cheating is impossible by
construction rather than by good intentions.

**Status: Phases 1–6 of 18 complete, plus Phases 7.1-7.3.1 — rules engine, information boundary, driver, baseline agents, search, the knowledge/belief layer, the information-aware search contract, determinized world generation, and single-world evaluation.**

- [`docs/architecture.md`](docs/architecture.md) — layers, invariants, design decisions
- [`docs/knowledge.md`](docs/knowledge.md) — observed vs deduced vs probable, the deduction rules, the belief model
- [`docs/search.md`](docs/search.md) — search architecture, the conservative abstraction, and its limits
- [`docs/agents.md`](docs/agents.md) — the agent contract, RandomBot, GreedyBot, RNG ownership
- [`docs/rules.md`](docs/rules.md) — the exact Podkidnoy rules implemented, and every variant ambiguity resolved
- [`docs/ai-design.md`](docs/ai-design.md) — which algorithms suit Durak, and why; strategy fusion; whether CFR is practical

## Install

Python 3.10+. The core engine has **no runtime dependencies**.

```bash
git clone <your-repo-url> durakfish
cd durakfish
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest                  # whole suite
pytest -v               # per-test names
pytest tests/test_deck.py -k fairness
```

Without installing, run from the repo root with `PYTHONPATH=src pytest`.
Hypothesis is optional; `tests/test_properties.py` skips itself when it
is missing.

## Playing a game

```python
from durakfish.ai import GreedyBot, RandomBot
from durakfish.simulation import play_game, summarize, verify_record

record = play_game([RandomBot("r"), GreedyBot("g")], seed=2026)
print(record)                 # GameRecord(seed=2026, r vs g, ... , P0 lost)
verify_record(record)         # replays it and checks the outcome

games = [play_game([RandomBot("r"), GreedyBot("g")], seed=s) for s in range(200)]
print(summarize(games).describe())    # greedy wins ~97% — the baseline floor
```

Agents receive an `InformationSet`, never a `GameState`. The hidden cards
are not reachable from what an agent holds, so cheating is a structural
impossibility rather than a rule agents are asked to follow. See
[`docs/rules.md`](docs/rules.md) §12 for the information model.

## What works today

```python
from durakfish import Card, Deck, Rank, Suit
from durakfish.cards import format_cards, sort_cards

deck = Deck.shuffled(seed=2026)      # seeded → fully reproducible
hand = deck.draw_up_to(6)
trump = deck.bottom_card             # face-up bottom card = trump in Durak

print(trump)                                        # 7♠
print(format_cards(sort_cards(hand, trump=trump.suit)))   # 8♣ 9♣ 10♣ 8♥ Q♥ K♠
print(len(deck))                                    # 30
```

Cards in three interchangeable representations:

```python
Card.parse("10H")            # Card('10H')   compact ASCII, case-insensitive
Card.parse("A♠")             # Card('AS')    unicode
Card.parse("TS")             # Card('10S')   'T' shorthand for ten
Card.of(Rank.QUEEN, Suit.DIAMONDS).unicode   # 'Q♦'
Card.from_code(51)           # dense integer code, 0..51
```

Cards are immutable, hashable and interned, so they work directly as set
members, dict keys and probability-table keys. `hash(card) == card.code`,
so a card doubles as an array index.

Decks support seeded shuffling, drawing (`draw`, `draw_up_to`), removal
for building analysis positions, `reset()`, `copy()`, and JSON-friendly
serialization via `to_list()` / `from_list()`.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Cards + deck | ✅ |
| 2 | Rules, trump logic, legal move generation, state and history | ✅ |
| 3 | Information boundary, agent protocol, driver, replay | ✅ |
| 4 | RandomBot, GreedyBot, agent contract | ✅ |
| 5 | Minimax, alpha-beta, SearchBot | ✅ |
| 6 | CardTracker, deduction, belief model | ✅ |
| 7.1 | Information-aware search contract | ✅ |
| 7.2 | Determinized world model | ✅ |
| 7.3.1 | Determinized world evaluation contract | ✅ |
| 7.3.2 | Multi-world aggregation | next |
| 6–7 | CardTracker, probability engine | |
| 8–9 | Perfect-information minimax, MCTS | |
| 10–11 | Determinization, ISMCTS | |
| 12–13 | Self-play, tournaments and Elo | |
| 14 | Profiling and optimisation | |
| 15–16 | Human CLI, GUI | |
| 17–18 | Computer vision, automated interaction | |

Phases are not skipped. Every phase ships with tests, and every strength
claim after Phase 13 is backed by an Elo measurement against previous
versions rather than by how the engine's explanations read.

## Honesty policy

The engine states what it does not know. Documented limitations —
strategy fusion under determinization, bias from heuristic rollout
cutoffs, the approximate nature of the opponent model, the absence of
bluffing — are recorded in [`docs/ai-design.md`](docs/ai-design.md) and
kept current rather than quietly dropped.

## License

MIT.
