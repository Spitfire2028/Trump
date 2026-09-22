"""What do the leaves ISMCTS evaluates actually look like?"""
import sys
sys.path[:0] = ["src", "."]
from durakfish.ai.evaluation import EVALUATORS, BaselineEvaluator
shape = {"n": 0, "nonempty_table": 0, "defending": 0, "defense_phase": 0,
         "undefended": 0, "applicable": 0}

class Probe(BaselineEvaluator):
    name = "probe_shape"
    def evaluate(self, node, ply=0):
        shape["n"] += 1
        shape["nonempty_table"] += bool(node.table)
        shape["defending"] += not node.searcher_is_attacker
        shape["defense_phase"] += node.phase.name == "DEFENSE"
        shape["undefended"] += any(not s.defended for s in node.table)
        shape["applicable"] += (
            BaselineEvaluator._facing_attacks(node)
            and any(not s.defended for s in node.table))
        return super().evaluate(node, ply)

EVALUATORS["probe_shape"] = Probe
import json, random
from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig
from durakfish.game.state import GameState
from durakfish.information import SearchInformation, observe
rows = [r for r in json.load(open("benchmarks/data/eval_sensitivity_corpus.json"))
        if r["legal_move_count"] >= 2][:40]
for row in rows:
    state = GameState.from_dict(row["state"]); player = row["player"]
    info = SearchInformation.from_view(observe(state, player))
    ISMCTS(ISMCTSConfig(iterations=200, evaluator="probe_shape")).search(
        info, random.Random(row["seed"] * 1000 + row["ply_index"]))
n = shape["n"]
for key, value in shape.items():
    if key == "n": continue
    print(f"  {key:<16} {value:>6} / {n}  ({100*value/n:.1f}%)")
