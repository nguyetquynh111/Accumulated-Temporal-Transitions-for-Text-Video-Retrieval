import json, csv, sys, statistics as st
from pathlib import Path
R = Path(sys.argv[1]); out = Path(sys.argv[2]); main = Path(sys.argv[3])
models = [("vjepa_baseline","B2 pooled"),("ablation_no_transition","no_transition"),
 ("ablation_transition_only","transition_only"),("ablation_shuffled_states","shuffled_states"),
 ("ablation_mean_transition","mean_transition"),("ablation_final_minus_initial","final_minus_initial"),
 ("ablation_full_transition_without_query_conditioning","B3 full_transition w/o query cond (E2 stage 1)"),
 ("transition_model","single-stage transition model"),
 ("two_stage_token_reranker_state_only","E1 (two-stage, states only)"),("two_stage_token_reranker","E2 (two-stage, [z, dz])")]
def load(stem, rev):
    alt = {"transition_model": "full_transition"}
    res = []
    for s in (42,43,44):
        for n in ([stem, alt.get(stem)] if rev else [stem]):
            p = R/f"{n}_seed{s}{'_reversed' if rev else ''}.json"
            if n and p.exists(): res.append(json.load(open(p))); break
    return res
ms = lambda xs: (f"{st.mean(xs):.4f}", f"{st.stdev(xs):.4f}" if len(xs)>1 else "")
cols = ["model","stem","seeds"]
for t in ("full","hard_negative","action_contrast"): cols += [f"{t}_R@1_mean",f"{t}_R@1_std",f"{t}_MRR_mean"]
cols += ["reversed_seeds","reversed_action_contrast_R@1_mean","reversed_action_contrast_R@1_std","reversed_minus_normal_AC_R@1"]
rows=[]
for stem,label in models:
    n = load(stem, False); r = load(stem, True)
    row = {"model":label,"stem":stem,"seeds":len(n)}
    for t in ("full","hard_negative","action_contrast"):
        row[f"{t}_R@1_mean"],row[f"{t}_R@1_std"] = ms([x[t]["recall@1"] for x in n])
        row[f"{t}_MRR_mean"] = ms([x[t]["mrr"] for x in n])[0]
    row["reversed_seeds"]=len(r)
    if r:
        a=[x["action_contrast"]["recall@1"] for x in r]
        row["reversed_action_contrast_R@1_mean"],row["reversed_action_contrast_R@1_std"]=ms(a)
        row["reversed_minus_normal_AC_R@1"]=f"{st.mean(a)-float(row['action_contrast_R@1_mean']):+.4f}"
    rows.append(row)
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,cols); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['model'][:46]:46s} n={r['seeds']} full={r['full_R@1_mean']} hard={r['hard_negative_R@1_mean']} AC={r['action_contrast_R@1_mean']}±{r['action_contrast_R@1_std']} rev={r.get('reversed_action_contrast_R@1_mean','-')} d={r.get('reversed_minus_normal_AC_R@1','-')}")
# reused results must match the originals
bad=0
for p in sorted(main.glob("*.json")):
    q=R/p.name
    if q.exists():
        a,b=json.load(open(p)),json.load(open(q))
        if a.get("action_contrast")!=b.get("action_contrast") or a.get("full")!=b.get("full"): bad+=1; print("CHANGED", p.name)
print("reused-result check:", "all identical" if not bad else f"{bad} changed")
