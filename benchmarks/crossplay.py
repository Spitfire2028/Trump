"""Cross-evaluator playing-strength study.  (Phase 7.3.11)

Diagnostic, not production. Nothing in the engine imports this.

The question is narrow: **when two agents share every search setting and
differ only in the named leaf evaluator, does either win more games?**

Phase 7.3.9 already established that evaluator substitution changes
12-36% of root decisions, and up to 75% inside ISMCTS. Phase 7.3.10
established that no candidate improves decision quality anywhere the
exact oracle can see -- which is almost entirely empty-talon positions,
where the candidates are provably identical to the baseline. This phase
asks the remaining question, and it is a different one: changing many
decisions is not the same as playing better, and this file exists to keep
those two claims apart.

Reused, not rebuilt
-------------------
The repository already has everything needed to play and record games:
``simulation.play_game`` (master-seed fan-out, per-seat redaction,
illegal-move refusal), ``simulation.GameRecord``, ``simulation.summarize``
and ``SearchBot``. None of it is modified. The only new component is an
``ISMCTSAgent`` adapter, because ISMCTS is a search class and no agent
wraps it -- and it lives here, in benchmarks, rather than in production.

No generalized tournament framework is built, and none should be.

EFFECT THRESHOLD -- declared before the final run
-------------------------------------------------
A win-rate difference is called **practically meaningful at 5 percentage
points** (e.g. 55% against 50%). Justification, fixed in advance:

* Phase 7.3.9 measured evaluator substitution changing 12-36% of root
  decisions. If changes on that scale move playing strength by less than
  5pp, the engineering conclusion is "equivalent", whatever a p-value
  says about 1pp.
* Phase 7.3.10 measured E3 at **3.34x** the baseline's evaluator cost.
  Anything under a few points of strength cannot justify that, so a
  difference too small to clear this bar cannot change a shipping
  decision either.

Smaller differences may be reported as observed; they are not claimed as
improvements. Larger differences without statistical support are reported
as inconclusive, not as improvements.

WHAT IS SHARED AND WHAT IS NOT
------------------------------
Common random numbers are applied where they are exact and honest about
where they are not:

* **Shared, exactly:** the deal. ``play_game`` derives ``deal_seed`` from
  the master seed before any agent acts, so the same seed always produces
  the same cards, trump and talon order regardless of which agents play.
* **Shared, exactly:** the seat pairing. Every seed is played twice, once
  in each orientation, so both evaluators face the identical deal from
  both seats.
* **Shared, nominally:** the per-agent RNG streams, also derived from the
  master seed.
* **NOT shared, and deliberately so:** ISMCTS internal trajectories after
  the first divergent decision. Once the evaluators pick different moves
  the games are different games; the streams remain the same *function*
  of the seed, but they are consumed differently. Forcing them to stay
  aligned would be artificial coupling, not variance reduction -- the
  distinction section 8 of the phase brief asks for.

Run: ``python benchmarks/crossplay.py <stage>`` with stage in
``pilot``, ``power``, ``deterministic``, ``ismcts``, ``analyse``,
``decisions``, ``oracle``.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.base import AgentBase
from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig
from durakfish.ai.search import SearchConfig
from durakfish.ai.search_bot import SearchBot
from durakfish.game import apply_move
from durakfish.information import SearchInformation
from durakfish.simulation import play_game

RESULTS = Path("benchmarks/data")

#: The Phase 7.3.10 family, by registry name. Verified at import.
FAMILY = ("baseline", "mechanistic", "structural", "flexible")

#: Declared above, before the final run.
MEANINGFUL_WIN_RATE_DIFFERENCE = 0.05

#: Strata, defined before any outcome was inspected. Game stage is by the
#: talon at the deciding moment; branching by mean legal-move count.
BRANCHING_BUCKETS = ((0.0, 3.0, "low"), (3.0, 5.0, "medium"), (5.0, 99.0, "high"))


class ISMCTSAgent(AgentBase):
    """Minimal Agent wrapper around the frozen ISMCTS search.

    ISMCTS is a search class, not an agent, and nothing in production
    wraps it. This adapter exists only so the existing game runner can
    drive it; it adds no policy of its own. The search, its configuration
    and its determinization model are untouched.
    """

    def __init__(self, config: ISMCTSConfig, name: str | None = None) -> None:
        super().__init__(name or f"ismcts-{config.evaluator}")
        self.config = config
        self.rng = random.Random(0)
        self.decisions = 0

    def reset(self, rng: random.Random) -> None:
        """Take the runner's per-seat stream. The search is stochastic, so
        unlike ``SearchBot`` this agent genuinely consumes it."""
        self.rng = rng
        self.decisions = 0

    def choose(self, view, legal):
        information = SearchInformation.from_view(view)
        result = ISMCTS(self.config).search(
            information, self.rng, root_moves=tuple(legal)
        )
        self.decisions += 1
        if result.move is None:  # pragma: no cover - search always returns
            return legal[0]
        return result.move


def make_agent(mode: str, evaluator: str, budget: int):
    """One agent. The ONLY thing that varies between arms is ``evaluator``."""
    if mode == "deterministic":
        return SearchBot(
            SearchConfig(depth=budget, evaluator=evaluator),
            name=f"search-d{budget}-{evaluator}",
        )
    if mode == "ismcts":
        return ISMCTSAgent(
            ISMCTSConfig(iterations=budget, evaluator=evaluator),
            name=f"ismcts-{budget}-{evaluator}",
        )
    raise SystemExit(f"unknown mode {mode!r}")


@dataclass
class GameOutcome:
    """One game, recorded per section 11 of the brief."""

    seed: int
    mode: str
    budget: int
    evaluator_p0: str
    evaluator_p1: str
    subject: str
    subject_seat: int
    winner: int | None
    durak: int | None
    draw: bool
    plies: int
    final_hand_sizes: tuple[int, ...]
    terminal_talon: int
    mean_branching: float
    seconds: float
    annotations: dict = field(default_factory=dict)

    @property
    def subject_score(self) -> float:
        """1 for a win, 0.5 for a draw, 0 for a loss, from the subject's seat."""
        if self.draw:
            return 0.5
        return 1.0 if self.winner == self.subject_seat else 0.0


def play_one(
    mode: str, budget: int, first: str, second: str, seed: int, subject_seat: int
) -> GameOutcome:
    """Play ``first`` as P0 against ``second`` as P1 on the given seed.

    The deal depends only on ``seed``, never on the agents, so the same
    seed with the seats reversed is the same cards from the other side.
    """
    agents = [make_agent(mode, first, budget), make_agent(mode, second, budget)]
    start = time.perf_counter()
    record = play_game(agents, seed=seed, observe_history=False)
    elapsed = time.perf_counter() - start

    # Replay to the terminal state for hand sizes and branching. This is a
    # diagnostic walk over ground truth and is never shown to an agent.
    state = record.initial_state
    branching = []
    for move in record.moves:
        from durakfish.game import get_legal_moves

        branching.append(len(get_legal_moves(state)))
        state = apply_move(state, move.move)

    subject = first if subject_seat == 0 else second
    return GameOutcome(
        seed=seed,
        mode=mode,
        budget=budget,
        evaluator_p0=first,
        evaluator_p1=second,
        subject=subject,
        subject_seat=subject_seat,
        winner=record.winner(),
        durak=record.durak,
        draw=record.is_draw,
        plies=record.length,
        final_hand_sizes=tuple(len(hand) for hand in state.hands),
        terminal_talon=len(state.talon),
        mean_branching=statistics.mean(branching) if branching else 0.0,
        seconds=elapsed,
    )


def paired_match(
    mode: str, budget: int, left: str, right: str, seeds: range | list
) -> list[dict]:
    """Every seed played in both orientations. Returns one row per PAIR.

    The pair -- not the game -- is the unit of analysis, because the two
    orientations share a deal and are therefore not independent.
    """
    rows = []
    for seed in seeds:
        as_p0 = play_one(mode, budget, left, right, seed, subject_seat=0)
        as_p1 = play_one(mode, budget, right, left, seed, subject_seat=1)
        rows.append(
            {
                "seed": seed,
                "mode": mode,
                "budget": budget,
                "left": left,
                "right": right,
                # Score for `left`, averaged over the two seats it played.
                "left_score": (as_p0.subject_score + as_p1.subject_score) / 2,
                "left_as_p0": as_p0.subject_score,
                "left_as_p1": as_p1.subject_score,
                "draws": int(as_p0.draw) + int(as_p1.draw),
                "plies": (as_p0.plies + as_p1.plies) / 2,
                "terminal_talon": (as_p0.terminal_talon + as_p1.terminal_talon) / 2,
                "mean_branching": (
                    as_p0.mean_branching + as_p1.mean_branching
                ) / 2,
                "seconds": as_p0.seconds + as_p1.seconds,
            }
        )
    return rows


# ======================================================================
# Statistics
# ======================================================================
def paired_bootstrap(
    values: list[float], null: float = 0.5, resamples: int = 10_000, seed: int = 7
) -> tuple[float, float, float, float]:
    """Resample PAIRS, never games and never moves.

    Returns (mean, lo, hi, two-sided p) with a 95% percentile interval.
    The p-value is the share of resamples on the far side of ``null``,
    doubled -- a bootstrap analogue of a two-sided test, reported
    alongside the interval rather than instead of it.
    """
    if not values:
        return (float("nan"),) * 4
    rng = random.Random(seed)
    n = len(values)
    observed = statistics.mean(values)
    if statistics.pstdev(values) == 0.0:
        # Degenerate: every pair scored identically. This happens by
        # construction for a self-pair of deterministic agents -- the same
        # deal with reversed seats is the same game, so the pair always
        # averages exactly 0.5. A bootstrap over identical values would
        # print p=0.000, which reads as a significant result and is the
        # opposite of the truth, so it is reported as no evidence.
        return observed, observed, observed, 1.0
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples) - 1]
    share = sum(1 for m in means if m <= null) / resamples
    p = 2 * min(share, 1 - share)
    return observed, lo, hi, min(p, 1.0)


def required_pairs(sd: float, effect: float = MEANINGFUL_WIN_RATE_DIFFERENCE) -> int:
    """Pairs needed to detect ``effect`` at 80% power, two-sided alpha 0.05.

    Standard paired formula: n = ((z_a + z_b) * sd / delta)^2, with the
    observed per-pair standard deviation from the pilot. Reported rather
    than assumed, because the whole point of the pilot is that the
    variance is measured instead of guessed.
    """
    if sd <= 0 or effect <= 0:
        return 0
    return math.ceil(((1.959964 + 0.841621) * sd / effect) ** 2)


# ======================================================================
# Stages
# ======================================================================
def pilot(_args: list[str]) -> None:
    """Measure the variance of the paired outcome before sizing anything."""
    print("pilot: measuring paired-outcome variance\n")
    print(f"  {'mode':<14} {'budget':>7} {'pair':<26} {'n':>4} "
          f"{'mean':>7} {'sd':>7} {'s/pair':>8} {'pairs for 5pp':>14}")
    plan = [
        ("deterministic", 3, ("baseline", "flexible")),
        ("deterministic", 3, ("baseline", "baseline")),
        ("ismcts", 50, ("baseline", "flexible")),
    ]
    for mode, budget, (left, right) in plan:
        seeds = range(40) if mode == "deterministic" else range(6)
        rows = paired_match(mode, budget, left, right, seeds)
        scores = [r["left_score"] for r in rows]
        sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        seconds = sum(r["seconds"] for r in rows) / len(rows)
        print(f"  {mode:<14} {budget:>7} {left + ' vs ' + right:<26} "
              f"{len(rows):>4} {statistics.mean(scores):>7.3f} {sd:>7.3f} "
              f"{seconds:>8.2f} {required_pairs(sd):>14}")
    print("\n  The self-pair (baseline vs baseline) is the control: its mean")
    print("  should sit at 0.5 and its sd is the irreducible game noise.")


def deterministic(args: list[str]) -> None:
    """Mode A: the full cross-play matrix under deterministic search."""
    pairs_wanted = int(args[0]) if args else 200
    budgets = (1, 3, 5)
    rows: list[dict] = []
    started = time.perf_counter()
    for budget in budgets:
        for i, left in enumerate(FAMILY):
            for right in FAMILY[i:]:
                rows.extend(
                    paired_match(
                        "deterministic", budget, left, right,
                        range(pairs_wanted),
                    )
                )
                print(f"  done d{budget} {left} vs {right} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    path = RESULTS / "crossplay_deterministic.json"
    path.write_text(json.dumps(rows))
    print(f"\n{len(rows)} pairs -> {path}")


def ismcts(args: list[str]) -> None:
    """Mode B: ISMCTS cross-play, budget-limited and honest about it."""
    pairs_wanted = int(args[0]) if args else 25
    budget = int(args[1]) if len(args) > 1 else 50
    rows: list[dict] = []
    started = time.perf_counter()
    # ISMCTS costs ~9s per seed-pair at 50 iterations, so the full 10-cell
    # matrix would need roughly four hours at any useful sample size. The
    # essential comparisons -- section 21 of the brief -- are each
    # candidate against the baseline, plus the baseline self-pair as a
    # control, so that is what is run. Cells not run are reported as not
    # run, never as equivalent.
    combos = [("baseline", name) for name in FAMILY]
    for left, right in combos:
        rows.extend(
            paired_match("ismcts", budget, left, right, range(pairs_wanted))
        )
        print(f"  done {budget}it {left} vs {right} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    path = RESULTS / f"crossplay_ismcts_{budget}.json"
    path.write_text(json.dumps(rows))
    print(f"\n{len(rows)} pairs -> {path}")


def _load(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS.glob(pattern)):
        rows.extend(json.loads(path.read_text()))
    return rows


def analyse(args: list[str]) -> None:
    """Sections 12-17: results, statistics, strata, all from saved pairs."""
    pattern = args[0] if args else "crossplay_*.json"
    rows = _load(pattern)
    if not rows:
        raise SystemExit(f"no results matching {pattern}")
    print(f"analysis over {len(rows)} recorded pairs "
          f"({2 * len(rows)} games)\n")

    modes = sorted({(r["mode"], r["budget"]) for r in rows})
    for mode, budget in modes:
        subset = [r for r in rows if r["mode"] == mode and r["budget"] == budget]
        label = f"{mode} budget={budget}"
        print(f"== {label}  ({len(subset)} pairs)")
        print(f"  {'pair':<28} {'n':>4} {'score':>7} {'95% CI':>17} "
              f"{'p':>7} {'P0 wr':>7} {'P1 wr':>7} {'draws':>7} {'verdict':>14}")
        # Iterate the pairs actually present, not a hard-coded family, so
        # results saved by a later phase are reported rather than skipped.
        present = sorted({(r["left"], r["right"]) for r in subset})
        for left, right in present:
            cell = [
                r for r in subset if r["left"] == left and r["right"] == right
            ]
            if not cell or left == right:
                continue
            if True:
                scores = [r["left_score"] for r in cell]
                mean, lo, hi, p = paired_bootstrap(scores)
                p0 = statistics.mean(r["left_as_p0"] for r in cell)
                p1 = statistics.mean(r["left_as_p1"] for r in cell)
                draws = sum(r["draws"] for r in cell) / (2 * len(cell))
                delta = abs(mean - 0.5)
                if delta >= MEANINGFUL_WIN_RATE_DIFFERENCE and p < 0.05:
                    verdict = "MEANINGFUL"
                elif p < 0.05:
                    verdict = "small"
                elif delta >= MEANINGFUL_WIN_RATE_DIFFERENCE:
                    verdict = "underpowered"
                else:
                    verdict = "equivalent"
                print(f"  {left + ' vs ' + right:<28} {len(cell):>4} "
                      f"{mean:>7.3f} [{lo:>6.3f},{hi:>6.3f}] {p:>7.3f} "
                      f"{p0:>7.3f} {p1:>7.3f} {draws:>7.3f} {verdict:>14}")
        # Self-pair controls.
        for name in sorted({r["left"] for r in subset} | {r["right"] for r in subset}):
            cell = [r for r in subset if r["left"] == name and r["right"] == name]
            if cell:
                scores = [r["left_score"] for r in cell]
                mean, lo, hi, p = paired_bootstrap(scores)
                print(f"  {'[control] ' + name + ' vs itself':<28} "
                      f"{len(cell):>4} {mean:>7.3f} [{lo:>6.3f},{hi:>6.3f}] "
                      f"{p:>7.3f}")
        print()


def strata(args: list[str]) -> None:
    """Sections 14-15: game stage and branching factor, buckets fixed first."""
    rows = _load(args[0] if args else "crossplay_*.json")
    baseline_pairs = [
        r for r in rows if r["left"] == "baseline" and r["right"] != "baseline"
    ]
    if not baseline_pairs:
        raise SystemExit("no baseline-vs-candidate pairs recorded")
    print(f"stratified baseline-vs-candidate results "
          f"({len(baseline_pairs)} pairs)\n")

    def report(title: str, key) -> None:
        print(f"  by {title}")
        buckets: dict = {}
        for row in baseline_pairs:
            buckets.setdefault(key(row), []).append(row["left_score"])
        for bucket in sorted(buckets, key=str):
            scores = buckets[bucket]
            mean, lo, hi, _ = paired_bootstrap(scores)
            flag = "" if len(scores) >= 30 else "   (n<30)"
            print(f"    {bucket!s:<14} n={len(scores):<5} "
                  f"baseline score {mean:>6.3f} [{lo:>6.3f},{hi:>6.3f}]"
                  f"{flag}")
        print()

    def branch_bucket(row: dict) -> str:
        value = row["mean_branching"]
        for low, high, name in BRANCHING_BUCKETS:
            if low <= value < high:
                return name
        return "high"

    report("game length", lambda r: "short (<40)" if r["plies"] < 40
           else "medium (40-70)" if r["plies"] < 70 else "long (70+)")
    report("terminal talon", lambda r: "empty" if r["terminal_talon"] == 0
           else "nonempty")
    report("mean branching factor", branch_bucket)
    report("budget", lambda r: f"{r['mode'][:4]}-{r['budget']}")


def decisions(args: list[str]) -> None:
    """Section 17: how often the evaluators actually disagree in play.

    Connects back to Phase 7.3.9: replays real game positions and asks
    each evaluator's agent to choose, counting disagreements. Positions
    come from games the pair actually played, not from a synthetic corpus.
    """
    budget = int(args[0]) if args else 3
    limit = int(args[1]) if len(args) > 1 else 40
    from durakfish.game import get_legal_moves
    from durakfish.information import observe

    print(f"root-decision disagreement in played games: depth {budget}, "
          f"{limit} games per pair\n")
    print(f"  {'pair':<28} {'positions':>10} {'differ':>8} {'rate':>7} "
          f"{'high-branch rate':>18}")
    for candidate in FAMILY[1:]:
        agents = {
            name: SearchBot(SearchConfig(depth=budget, evaluator=name))
            for name in ("baseline", candidate)
        }
        seen = differ = high_seen = high_differ = 0
        for seed in range(limit):
            record = play_game(
                [make_agent("deterministic", "baseline", budget),
                 make_agent("deterministic", candidate, budget)],
                seed=seed, observe_history=False,
            )
            state = record.initial_state
            for move in record.moves:
                legal = get_legal_moves(state)
                player = state.current_player
                if player is not None and len(legal) >= 2:
                    view = observe(state, player)
                    picks = {
                        name: agent.choose(view, legal)
                        for name, agent in agents.items()
                    }
                    seen += 1
                    disagreed = len(set(picks.values())) > 1
                    differ += disagreed
                    if len(legal) >= 5:
                        high_seen += 1
                        high_differ += disagreed
                state = apply_move(state, move.move)
        rate = 100 * differ / seen if seen else 0.0
        high = 100 * high_differ / high_seen if high_seen else 0.0
        print(f"  {'baseline vs ' + candidate:<28} {seen:>10} {differ:>8} "
              f"{rate:>6.1f}% {high:>17.1f}%")
    print("\n  Decision disagreement is not strength. Phase 7.3.9 established")
    print("  the first; this phase's win rates address the second.")


def oracle(args: list[str]) -> None:
    """Section 19: sanity-check played positions against the exact solver.

    Not a strength metric -- Phase 7.3.10 established that the solver's
    reachable domain is almost entirely empty-talon, where the candidates
    are identical to the baseline by construction. This only checks that
    agents are not playing oracle-losing moves where the oracle can see.
    """
    import time as _time

    budget = int(args[0]) if args else 3
    seconds = float(args[1]) if len(args) > 1 else 180.0
    sys.setrecursionlimit(60_000)
    from talon_reference import solve

    from durakfish.game import get_legal_moves
    from durakfish.information import observe

    print(f"exact cross-check on played positions (depth {budget})\n")
    print(f"  {'evaluator':<14} {'solved positions':>17} {'oracle-optimal':>16}")
    for name in FAMILY:
        agent = SearchBot(SearchConfig(depth=budget, evaluator=name))
        solved = optimal = 0
        start = _time.time()
        for seed in range(60):
            if _time.time() - start > seconds:
                break
            record = play_game(
                [make_agent("deterministic", name, budget),
                 make_agent("deterministic", name, budget)],
                seed=seed, observe_history=False,
            )
            state = record.initial_state
            for move in record.moves:
                if _time.time() - start > seconds:
                    break
                legal = get_legal_moves(state)
                player = state.current_player
                if player is not None and len(legal) >= 2:
                    truth = {}
                    ok = True
                    for option in legal:
                        result = solve(
                            apply_move(state, option), player, budget=120_000
                        )
                        if not result.exact:
                            ok = False
                            break
                        truth[option] = result.value
                    if ok:
                        solved += 1
                        chosen = agent.choose(observe(state, player), legal)
                        optimal += truth[chosen] == max(truth.values())
                state = apply_move(state, move.move)
        rate = 100 * optimal / solved if solved else 0.0
        print(f"  {name:<14} {solved:>17} {rate:>15.1f}%")


def performance(args: list[str]) -> None:
    """Section 22: the cost of each evaluator in actual games."""
    budget = int(args[0]) if args else 3
    games = int(args[1]) if len(args) > 1 else 20
    print(f"cost in play: deterministic depth {budget}, {games} games each\n")
    print(f"  {'evaluator':<14} {'s/game':>9} {'games/hour':>12} "
          f"{'mean plies':>11} {'vs baseline':>12}")
    reference = None
    for name in FAMILY:
        start = time.perf_counter()
        plies = 0
        for seed in range(games):
            record = play_game(
                [make_agent("deterministic", name, budget),
                 make_agent("deterministic", name, budget)],
                seed=seed, observe_history=False,
            )
            plies += record.length
        elapsed = time.perf_counter() - start
        per_game = elapsed / games
        reference = reference or per_game
        print(f"  {name:<14} {per_game:>9.3f} {3600 / per_game:>12,.0f} "
              f"{plies / games:>11.1f} {per_game / reference:>11.2f}x")


#: Phase 7.3.12: the unrepaired candidate and its four repairs.
REPAIR_FAMILY = ("flexible", "flex_r1", "flex_r2", "flex_r3", "flex_r4")


def repairs(args: list[str]) -> None:
    """Phase 7.3.12 -- baseline against R0 and each repair, depths 1/3/5.

    Reuses this module's paired design unchanged: same deals, both seat
    orientations, pair as the unit of analysis. Only the opponent's
    evaluator name differs from the Phase 7.3.11 runs.
    """
    pairs_wanted = int(args[0]) if args else 300
    rows: list[dict] = []
    started = time.perf_counter()
    for budget in (1, 3, 5):
        for candidate in REPAIR_FAMILY:
            rows.extend(
                paired_match(
                    "deterministic", budget, "baseline", candidate,
                    range(pairs_wanted),
                )
            )
            print(f"  done d{budget} baseline vs {candidate} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    path = RESULTS / "crossplay_repairs.json"
    path.write_text(json.dumps(rows))
    print(f"\n{len(rows)} pairs -> {path}")


def repair_pilot(args: list[str]) -> None:
    """Variance of the repaired-candidate comparison, before sizing."""
    pairs_wanted = int(args[0]) if args else 40
    print("repair pilot: paired-outcome variance, depth 3\n")
    print(f"  {'pair':<30} {'n':>4} {'mean':>7} {'sd':>7} "
          f"{'pairs for 5pp':>14}")
    for candidate in REPAIR_FAMILY:
        rows = paired_match(
            "deterministic", 3, "baseline", candidate, range(pairs_wanted)
        )
        scores = [r["left_score"] for r in rows]
        sd = statistics.pstdev(scores)
        print(f"  {'baseline vs ' + candidate:<30} {len(rows):>4} "
              f"{statistics.mean(scores):>7.3f} {sd:>7.3f} "
              f"{required_pairs(sd):>14}")


STAGES = {
    "pilot": pilot,
    "repairs": repairs,
    "repair_pilot": repair_pilot,
    "deterministic": deterministic,
    "ismcts": ismcts,
    "analyse": analyse,
    "strata": strata,
    "decisions": decisions,
    "oracle": oracle,
    "performance": performance,
}


def main() -> None:
    from durakfish.ai.evaluation import EVALUATORS

    missing = [name for name in FAMILY if name not in EVALUATORS]
    if missing:
        raise SystemExit(
            f"registry inconsistency: {missing} not registered. Stopping "
            "rather than modifying the candidates."
        )
    args = sys.argv[1:]
    stage = args[0] if args else "pilot"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    STAGES[stage](args[1:])


if __name__ == "__main__":
    main()
