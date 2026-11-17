"""Table of the é-hyphen spelling test (run_spell.sh): per model, word and written form, P and Carbon right of 8.

    TESTS=spell_tests.json V3DATA=... python3 spell_report.py <spell dir>     # after v3_ana.py acro and carbon_judge.py run
"""
import json, sys

from acro import grade
from carbon_judge import gphones
from v3_ana import tests, word_ok_g

E = sys.argv[1].rstrip("/")
T = tests()
cell = {}
for r in json.load(open(f"{E}/graded.json")):
    cell.setdefault((r["model"], r["id"]), {"p": [], "c": []})["p"].append((r["p"], r["p_heard"]))
for k, h in json.load(open(f"{E}/carbon.json")).items():
    m, base = k.split("/")
    c = T[base[:-4].rsplit("_", 1)[0]]
    ok = word_ok_g(c["token"], h) if c["kind"] == "word" else bool(h) and grade(gphones(h), c["targets"][0], [], [], "id") != "wrong"
    cell.setdefault((m, c["id"]), {"p": [], "c": []})["c"].append((ok, h))
models = sorted({m for m, _ in cell})
L = ["| word | written | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
for tid, c in T.items():
    row = []
    for m in models:
        x = cell.get((m, tid), {"p": [], "c": []})
        bad_p = [h for ok, h in x["p"] if not ok]
        bad_c = [h for ok, h in x["c"] if not ok]
        row.append(f"P {sum(ok for ok, _ in x['p'])}/{len(x['p'])}" + (f" `{max(set(bad_p), key=bad_p.count)}`" if bad_p else "")
                   + f"<br>C {sum(ok for ok, _ in x['c'])}/{len(x['c'])}" + (f" `{max(set(bad_c), key=bad_c.count)}`" if bad_c else ""))
    L.append(f"| {c['token']} | {c['token'] if c.get('form') == 'raw' else c['form']} | " + " | ".join(row) + " |")
print("\n".join(L))
