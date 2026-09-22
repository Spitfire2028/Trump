# Agents

Phase 4 adds the first real policies. This document is the contract they
satisfy and that every later agent — search engines included — will
satisfy too.

## The contract

```python
class Agent(Protocol):
    name: str
    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move: ...
    def reset(self, rng: random.Random) -> None: ...
```

Unchanged from Phase 3, and it stays unchanged. Specifically:

- **No `GameState` argument, ever.** An agent receives a redacted view.
  The hidden cards are not reachable from anything it holds, so cheating
  is structurally impossible rather than a rule agents are trusted to
  follow. `tests/test_phase4_audit.py` walks the whole object graph
  reachable from a bot's arguments and asserts no `GameState`,
  `HistoryNode` or `GameEvent` appears and that every reachable card has
  been observed.
- **No hidden-state side channels.** No optional arguments, no callbacks
  into the runner, no access to the rules engine.
- **The return value must be one of `legal`.** Anything else raises
  `IllegalAgentMoveError`. The driver never repairs an agent's mistake:
  silently substituting a legal move would corrupt every statistic
  gathered from that agent.

### Errors

| Situation | Behaviour |
|---|---|
| `legal` is empty | `AgentError`. The engine never offers an empty choice in a live position, so this means the caller is broken. |
| Agent returns a move outside `legal` | `IllegalAgentMoveError`, naming the agent. Deterministic and identical on repeat runs. |
| `legal` contains duplicates | Tolerated; the bot still returns something from the list. A malformed list is the caller's bug, and the bot's job is only to stay inside it. |

## RNG ownership

The simulation layer owns randomness. One master seed fans out:

```
seed ──► deal_seed
     └─► agent_seeds[0], agent_seeds[1], ...
```

Each agent receives its own `random.Random` through `reset()`. Agents
**never** call the global `random` module — there is a test that
monkey-patches every global entry point to raise and then plays a full
game.

Separate streams mean an agent's randomness cannot perturb the deal, so
two bots can be compared on identical cards. The Phase 3 derivation is
unchanged; Phase 4 added nothing to it.

`reset(rng)` must establish the *complete* per-game state. A reset agent
is indistinguishable from a fresh one, which is what lets one agent
object be reused across thousands of games without leaking state between
them.

## RandomBot

Chooses uniformly from the legal moves. It is the floor every later agent
must beat: an engine that cannot outscore uniform random play is not
doing anything.

- Draws only from its injected generator.
- **Order independent.** It sorts into canonical order before indexing,
  so "uniform over the legal moves" is a statement about the *set*. The
  same moves in a different order give the same answer from the same RNG
  state.
- Ignores the view entirely. That makes it trivially immune to
  information leaks, and means a later bot beating it is a claim about
  *using* information, not about having more of it.

## GreedyBot

Deliberately simple, fully deterministic, and not a strong player. One
idea, applied everywhere:

> **Spend the cheapest card you can, and never spend a trump when
> spending anything was optional.**

Cost is the pair `(is_trump, code)`: every non-trump is cheaper than
every trump, and within a group lower cards are cheaper.

| Situation | Rule |
|---|---|
| Opening a bout | Attacking is compulsory — play the cheapest card, trump or not. |
| Adding to a bout, or throwing in after a take | Add the cheapest card, **unless it is a trump**, in which case end the bout. Junk is worth throwing in; trumps are not. |
| Defending | Beat the attack with the cheapest card that does the job. Never take voluntarily. |

### Known weaknesses, kept on purpose

"Never take voluntarily" is plainly wrong in real play: GreedyBot will
beat a six with the ace of trumps rather than pick up one card. It has no
model of the opponent, no lookahead, and no awareness of the endgame. A
baseline whose weaknesses are obvious is more useful than one whose
weaknesses are subtle — when a Phase 8+ search engine beats it, the
reason should be legible.

It wins roughly 97–98% against RandomBot from either seat, which is a
sanity check on the heuristic, not a strength claim.

### Determinism and tie-breaking

GreedyBot ignores its RNG. **The master seed changes the cards dealt but
never GreedyBot's reaction to them**, so two GreedyBots seeded
differently play identically. That is expected, not a defect; only
RandomBot's trajectory responds to the agent seed.

Every comparison ends with `canonical_move_key` — move type, then card
codes — so ordering never depends on object identity, hashing, or the
order the engine emitted moves in. Permuting `legal` cannot change the
decision.

One honest note: that tie-breaker is currently **defensive rather than
exercised**. Two card-carrying moves cannot tie, because `card_cost` is
injective over the deck; and `TakeCards` and `EndAttack` are never legal
in the same position (0 occurrences in 22,038 sampled positions). A
mutation replacing it with `id()` passes the entire suite — it is
unreachable, not untested. It stays because a scoring function with
coarser buckets would make ties reachable immediately, and discovering
that through a non-reproducible bot would be far worse than carrying a
few unused lines.

## Statistics

`durakfish.simulation.summarize` aggregates finished records: games,
plies, decisions, wins, losses, draws. It reads records after the fact,
so collecting statistics cannot perturb a game.

It is deliberately too small to grow into a tournament framework by
accident. Ratings, round-robins and significance testing are Phase 13.

## Deliberately not implemented

No card tracking, no opponent inference, no probabilities, no search of
any kind, no evaluation function intended for search, no learning, no
self-play batches, no tournaments or ratings, no parallelism, no
interface. Both bots receive exactly what a future search engine will
receive — an `InformationSet` and a list of legal moves — and neither
infers a single hidden card.

## Performance

| | Latency | Rate |
|---|---|---|
| `RandomBot.choose()` | 1.8 us | 559,000/s |
| `GreedyBot.choose()` | 2.8 us | 353,000/s |
| `observe()` with history | 186 us | 5,400/s |

The agent layer is about **1% of the cost of a decision**. The
information boundary remains the bottleneck, as it was at the end of
Phase 3, and the fix for it is still Phase 6's incremental tracker.
Optimising the bots would be optimising the wrong thing.
