# Which algorithms actually suit Durak

This document answers spec §62, §63 and §75 before any search code is
written, because the choice of algorithm determines the shape of
`GameState` and `InformationSet` — and those are Phase 2–3 decisions.

## 1. What kind of game Durak is

Four properties matter more than any algorithm choice:

1. **Imperfect information.** Opponents' hands are hidden. This is a
   different problem from randomness, and conflating the two is the
   single most common mistake in game-AI projects.
2. **Stochastic.** Draws from the talon are random, so even with all
   hands revealed the future is not determined.
3. **Sequentially revealing.** Every attack, defence and pickup exposes
   real cards. Uncertainty shrinks fast and monotonically.
4. **It becomes a perfect-information game.** This is the key structural
   insight for DurakFish, and it is worth stating loudly:

> In two-player Durak, once the talon is empty, the opponent's hand is
> *exactly deducible*: it is the 36 cards minus your hand, minus the
> discard pile, minus the table. There is no hidden information left.

So Durak is not one game. It is an imperfect-information game that
converts into a **finite, deterministic, perfect-information game** —
usually with twelve or fewer cards in play — precisely when the outcome
is decided. Very few card games hand you that. It means the endgame is
not a place for better heuristics; it is a place for an *exact solver*,
and exact play there is worth more than any mid-game refinement.

(With three or more players the conversion is partial: you deduce the
*union* of the opponents' hands but not the split. The endgame solver
must therefore be written against 2-player assumptions and gated on
player count, not silently reused.)

## 2. Candidate algorithms

**Minimax / alpha-beta.** Assumes perfect information. Run on the true
game state it is straightforward cheating, and its evaluation of a
position it cannot legally see is meaningless. Two legitimate uses:
(a) a debug oracle for testing move generation and evaluation, and
(b) the real, exact **endgame solver** once the talon is empty. Not a
mid-game answer.

**Expectiminimax.** Adds chance nodes for the talon draws. This handles
*stochasticity*, not *hidden information*. Modelling the opponent's
unknown hand as a chance node at the root is mathematically the same as
averaging over clairvoyant play, which is exactly the failure described
in §4 below. Useful once inside a determinized world; not a solution.

**MCTS (UCT).** Anytime, no evaluation function required, handles high
branching well. But plain MCTS assumes it is searching a single concrete
state, so it needs to be told which world it is in. It is a component,
not the answer.

**Determinization / PIMC.** Sample a complete hidden state consistent
with what you know, solve it as a perfect-information game, repeat,
vote. Empirically much stronger than its theory deserves in trick-taking
games — GIB in bridge, and the strongest Skat and Hearts programs work
this way. Long et al. (2010) identified when PIMC does well: high
*disambiguation* (uncertainty resolves quickly), low *leaf correlation*,
moderate bias. **Durak scores well on disambiguation** — information is
revealed constantly and the endgame is fully revealed — which is a real
reason to expect PIMC-family methods to work here. Its flaws are §4.

**ISMCTS.** One tree whose nodes are *information sets*, re-determinized
each iteration, with statistics shared across sampled worlds. Because
each information set carries a single set of statistics, the search
cannot commit to different actions in different worlds at the same
information set — which is precisely the strategy-fusion failure that
PIMC suffers from. This is the best practical fit for DurakFish's
mid-game.

**CFR / MCCFR.** Converges toward Nash equilibrium in two-player
zero-sum imperfect-information games, which Durak essentially is.
Feasibility is discussed in §6: the honest answer is *not for full
36-card Durak in this project's near term*.

**Deep CFR / ReBeL / Player of Games.** The research-grade route:
function approximation over information sets plus search. Real, but
requires a fast, correct, well-tested simulator to generate data — which
is exactly what Phases 1–13 build. Attempting it first is how projects
die.

**AlphaZero-style self-play.** Assumes perfect information. Its direct
transplant to Durak is unsound; the imperfect-information analogue is
the ReBeL/PoG line above.

## 3. The chosen architecture

```
CardTracker            exact deduction — what is certain
      ↓
OpponentModel          belief over hidden cards — what is likely
      ↓
weighted determinization ─→ ISMCTS ─→ heuristic evaluation at the cutoff
      ↓
ExactEndgameSolver     alpha-beta, used the moment the talon is empty
```

This is broadly the combination the spec proposed, and I endorse it —
with one change of emphasis. The spec lists exact endgame search last,
as a refinement. It should be **built early and treated as central**,
because §1 shows the endgame is genuinely solvable rather than merely
better-approximated. A mediocre mid-game plus perfect endgame beats a
good mid-game plus sloppy endgame in this specific game.

Order of construction: hard deduction before soft belief, belief before
sampling, sampling before search. Each layer is testable on its own,
which is the whole point of the phase plan.

## 4. Strategy fusion, and why it is not solved by ISMCTS

**Strategy fusion.** A determinized search sees a world in which the
opponent holds A♠ and plays the line that refutes A♠; in the next sample
it sees a world without A♠ and plays a different line. Averaging the
results implicitly assumes the AI will *know which world it is in* when
the moment comes. It will not. The result is an engine that overvalues
lines requiring knowledge it does not have.

ISMCTS attacks this directly: statistics live on information sets, so a
single policy is learned per information set and the search cannot
"decide later" per world.

**But ISMCTS does not eliminate the problem**, and the project should
not pretend otherwise:

- *Opponent-side fusion remains.* Basic (single-observer) ISMCTS treats
  the opponent as if it, too, could see the determinized world. The
  opponent therefore plays too well and too knowledgeably. Multiple-
  Observer ISMCTS models each player's information set separately and
  fixes much of this, at real complexity cost. Plan: SO-ISMCTS first,
  MO-ISMCTS as a measured upgrade with an Elo test attached.
- *Non-locality is untouched.* A perfect-information search assumes any
  reachable position is reachable; in reality an opponent playing well
  would never have let certain positions arise with certain hands. The
  belief distribution should encode this, which is why the opponent
  model matters as much as the search does.
- *Sampling bias.* If the opponent model is wrong, the sampled worlds
  are wrong, and more simulations converge more confidently on a worse
  answer. Simulation count is not a substitute for calibration.

**Consequence for the roadmap:** the opponent model needs its own
correctness tests (impossible cards must have probability zero; sampled
hands must respect known hand size and every observed constraint), and
strength must be measured by Elo against previous versions, not by
"looks reasonable".

## 5. Where exact search applies

Trigger the endgame solver when: two players, talon empty, and total
cards in play below a configured threshold. Since the position is then
perfect information, alpha-beta with a transposition table returns the
*true* result: win, loss, or draw with best play. There is no evaluation
function, no sampling, and no error.

Practical notes: branching in Durak comes from multi-card attacks and
the defender's choices; the state at that point is small (two hands plus
a table), and states repeat heavily across move orders, so a
transposition table pays off unusually well. Complexity grows sharply
with hand size, so the threshold must be measured, not guessed — that
is what the Phase 14 benchmarks are for.

## 6. Is CFR practical for Durak?

**Honest answer: not for the full game, not soon.**

*Scale.* The initial deal alone gives C(36,6) × C(30,6) ≈ 1.15 × 10¹²
distinct two-player deals. Information sets multiply that by every
distinct public history — attacks, defences, pickups, draws — of variable
length. Tabular CFR must store regret and strategy vectors per
information set and revisit them many times. This is orders of magnitude
beyond poker-scale tabular solving, and Durak has none of poker's
convenient betting-round structure to abstract along.

*Information-set representation.* An information set would be: own hand
(canonicalised by suit isomorphism where the trump suit permits it),
public history compressed to (cards discarded, cards on the table,
opponent hand size, talon size, trump), and the current bout position.
Suit isomorphism is the one large, sound abstraction available — three
non-trump suits are interchangeable up to relabelling, giving roughly a
6× reduction. Helpful; nowhere near sufficient.

*Actions.* Structured move objects map cleanly to CFR actions
(`Attack(card)`, `Defend(attacked, with)`, `Take`, `EndAttack`), which is
one more reason Phase 2 uses typed moves rather than strings.

*Storage.* MCCFR (outcome or external sampling) with a hashed
information-set key, or Deep CFR with a neural advantage network to avoid
tabular storage entirely.

*Would abstraction be required?* Yes, heavily — and Durak's abstraction
options are poor. Bucketing hands by "strength" is far lossier here than
in poker, because a hand's value depends on the specific ranks held
relative to what remains, not on a scalar strength.

**Recommendation.** Do not implement CFR in the first strong version.
Keep the door open cheaply: typed actions, an `InformationSet` with a
canonical hashable key, and an `Agent` interface that a CFR-trained
policy could satisfy. If CFR is explored later, do it on a reduced
variant (20- or 24-card Durak) where a tabular solve is tractable, and
use the resulting near-equilibrium policy as a *benchmark opponent* to
measure how exploitable the ISMCTS engine is. That is worth more than a
half-abstracted CFR attempt on the full game.

## 7. Known limitations, stated up front

- Heuristic evaluation at a rollout cutoff introduces bias that more
  simulations will not remove.
- The baseline opponent model is a constrained uniform distribution. It
  is an approximation, not a belief update from a model of opponent
  reasoning, and it will misprice a strong human's deliberate choices.
- Behavioural inference ("the opponent declined to beat 9♠, so they
  probably lack ≥10♠ and non-trump spades") is only sound against an
  opponent who plays well. Against a weak or deliberately deceptive
  opponent it can be worse than uniform. Constraints derived from
  *legality* are always safe; constraints derived from *choice* are not,
  and the two must be stored separately.
- Nothing here bluffs. Deliberately concealing information has value in
  Durak, and determinization-family methods do not discover it.

None of these are reasons not to build the engine. They are the reasons
to measure it by Elo against its own previous versions rather than by
how convincing its explanations sound.
