"""Is V4 actually consulted inside ISMCTS, and does it ever differ there?"""
import sys
sys.path[:0] = ["src", "."]
from durakfish.ai.evaluation import EVALUATORS, EvaluationWeights, BaselineEvaluator

calls = {"n": 0, "differing": 0}
base = BaselineEvaluator()

class Probe(BaselineEvaluator):
    name = "probe_v4"
    def __init__(self):
        super().__init__(EvaluationWeights(open_obligation=3.0))
    def evaluate(self, node, ply=0):
        value = super().evaluate(node, ply)
        calls["n"] += 1
        if abs(value - base.evaluate(node, ply)) > 1e-9:
            calls["differing"] += 1
        return value

EVALUATORS["probe_v4"] = Probe

import json, random
from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig
from durakfish.game.state import GameState
from durakfish.information import SearchInformation, observe

rows = [r for r in json.load(open("benchmarks/data/eval_sensitivity_corpus.json"))
        if r["legal_move_count"] >= 2][:40]
for row in rows:
    state = GameState.from_dict(row["state"])
    player = row["player"]
    info = SearchInformation.from_view(observe(state, player))
    seed = row["seed"] * 1000 + row["ply_index"]
    ISMCTS(ISMCTSConfig(iterations=200, evaluator="probe_v4")).search(
        info, random.Random(seed))
print(f"evaluator calls inside ISMCTS: {calls['n']}")
print(f"  calls where V4 differed from the baseline: {calls['differing']} "
      f"({100*calls['differing']/max(calls['n'],1):.2f}%)")
