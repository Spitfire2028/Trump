import sys
sys.path[:0] = ["src", "."]
from durakfish.ai.evaluation import EVALUATORS, BaselineEvaluator
from collections import Counter
pairs = Counter(); c = {"n":0, "facing":0, "facing_and_undef":0}

class Probe(BaselineEvaluator):
    name = "probe2"
    def evaluate(self, node, ply=0):
        c["n"] += 1
        facing = BaselineEvaluator._facing_attacks(node)
        c["facing"] += facing
        c["facing_and_undef"] += facing and any(not s.defended for s in node.table)
        pairs[(node.phase.name, "defender" if not node.searcher_is_attacker
               else "attacker", "to_move=" + node.to_move.name)] += 1
        return super().evaluate(node, ply)

EVALUATORS["probe2"] = Probe
import json, random
from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig
from durakfish.game.state import GameState
from durakfish.information import SearchInformation, observe
rows = [r for r in json.load(open("benchmarks/data/eval_sensitivity_corpus.json"))
        if r["legal_move_count"] >= 2][:40]
for row in rows:
    state = GameState.from_dict(row["state"]); player = row["player"]
    info = SearchInformation.from_view(observe(state, player))
    ISMCTS(ISMCTSConfig(iterations=200, evaluator="probe2")).search(
        info, random.Random(row["seed"]*1000 + row["ply_index"]))
print("leaves:", c["n"], " _facing_attacks:", c["facing"],
      " facing+undefended:", c["facing_and_undef"])
for key, count in pairs.most_common(8):
    print(f"  {key}  {count}")
