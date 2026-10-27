"""Aggregate toy results: per (camera, variant) mean over seeds and paired delta vs a reference variant.
usage: python aggregate.py results.jsonl pe10"""
import json, sys
from collections import defaultdict

path, ref = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path) if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[(r['camera'], r['variant'])][r['seed']] = r
for cam in ('teleport', 'smooth'):
    print(f"\n== camera: {cam} ==   (delta = paired difference vs {ref}, same seed)")
    print(f"{'variant':24s} {'n':>2s} {'train':>6s} {'fixed':>6s} {'near':>6s} {'d_fixed':>8s} {'d_near':>8s} {'gap':>6s} {'rough':>6s}")
    refd = by.get((cam, ref), {})
    for (c, v), d in sorted(by.items()):
        if c != cam:
            continue
        n = len(d)
        m = lambda k: sum(x[k] for x in d.values()) / n
        common = [s for s in d if s in refd]
        dd = lambda k: (sum(d[s][k] - refd[s][k] for s in common) / len(common)) if common else float('nan')
        rng = lambda k: (min(d[s][k] - refd[s][k] for s in common), max(d[s][k] - refd[s][k] for s in common)) if common else (0, 0)
        lo, hi = rng('test_near' if cam == 'smooth' else 'test_fixed')
        print(f"{v:24s} {n:2d} {m('train'):6.2f} {m('test_fixed'):6.2f} {m('test_near'):6.2f} {dd('test_fixed'):+8.2f} {dd('test_near'):+8.2f} "
              f"{m('train') - m('test_fixed'):6.2f} {m('rough'):6.2f}   [{lo:+.2f},{hi:+.2f}]")
