import sys; sys.path.insert(0,'.')
sys.path.insert(0,'tests/redteam')
from test_adaptive import _ScoreClassifier
from redteam.adaptive.oracle import CachingClassifier
from redteam.adaptive.goals import enumerate_goals
from redteam.adaptive.search import run_goal
goal=enumerate_goals()[0]
def improved(res):
    per={}
    for q in res.queries: per.setdefault(q["core_id"],[]).append(q)
    for qs in per.values():
        seen=[q["best_score_after"] for q in qs if q["best_score_after"] is not None]
        if len(seen)>=2 and min(seen[1:])<seen[0]: return True
    return False
for sf in (True, False):
    res=run_goal(goal, CachingClassifier(_ScoreClassifier()), budget=40, score_feedback=sf, seed=3)
    print('score_feedback',sf,'test condition "improved" =',improved(res))
