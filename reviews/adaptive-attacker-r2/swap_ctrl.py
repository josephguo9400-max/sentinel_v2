"""Score matched controls: each committed success body with the reserved target swapped for the
scenario's authorised address (identical carrier, placement, framing, mutation)."""
import json, sys
sys.path.insert(0, '.')
from redteam.adaptive.goals import select_goals
from redteam.adaptive.oracle import make_gateway_classifier, decider_score
from redteam.adaptive.search import _authorised_address
raw, out = sys.argv[1], sys.argv[2]
goals = {g.goal_id: g for g in select_goals('all')}
clf = make_gateway_classifier('http://127.0.0.1:8000')
res = []
for l in open(raw):
    r = json.loads(l)
    if not (r['classifier_allowed'] and r['is_harmful']):
        continue
    g = goals[r['goal_id']]
    addr = _authorised_address(g.scenario())
    for kind, body in (('orig', r['body']), ('swap_authorised', r['body'].replace(g.target, addr))):
        v = clf.classify(body, 'rv')
        res.append({'goal_id': r['goal_id'], 'core_id': r['core_id'], 'kind': kind,
                    'body_sha16': r['body_sha16'], 'decision': v.decision, 'score': decider_score(v)})
json.dump(res, open(out, 'w'), indent=1)
print('n', len(res))
