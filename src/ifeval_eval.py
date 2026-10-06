"""확장 실험 E2: 공개 평가 IFEval 541문항(blueprint 10.2).

    python src/ifeval_eval.py generate --cond B0|B1|M1   ->  results/ifeval/gen_<cond>.jsonl
    python src/ifeval_eval.py report                     ->  results/ifeval/metrics.json

판정은 Google Research 공식 코드(src/instruction_following_eval, Apache-2.0, 수정 없음)가 한다.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from evaluate import load_jsonl, mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ifeval/input_data.jsonl"
MAX_NEW = 512
OVERLAP = {"punctuation:no_comma", "change_case:english_lowercase", "detectable_format:number_bullet_lists",
           "detectable_format:json_format", "startend:end_checker", "startend:quotation", "keywords:frequency",
           "length_constraints:number_words"}


def generate_cond(cond):
    from lm import generate, load
    tok, model = load(adapter_dir=ROOT / "adapter" if cond == "M1" else None)
    items = load_jsonl(DATA)
    t0 = time.time()
    outs = generate(tok, model, [it["prompt"] for it in items], strict=(cond == "B1"), max_new_tokens=MAX_NEW, batch_size=16,
                    on_batch=lambda d, n: print(f"\r{cond} ifeval {d}/{n} {time.time() - t0:.0f}s", end="", flush=True))
    print()
    rows = []
    for it, o in zip(items, outs):
        n_tok = len(tok(o[0], add_special_tokens=False)["input_ids"])
        rows.append({"key": it["key"], "prompt": it["prompt"], "response": o[0], "tokens": n_tok})
    out = ROOT / f"results/ifeval/gen_{cond}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def report():
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from instruction_following_eval import evaluation_lib as E
    inputs = E.read_prompt_list(str(DATA))
    res, vec = {}, {}
    for cond in ["B0", "B1", "M1"]:
        path = ROOT / f"results/ifeval/gen_{cond}.jsonl"
        if not path.exists():
            continue
        gens = load_jsonl(path)
        p2r = {g["prompt"]: g["response"] for g in gens}
        r = {}
        for mode, fn in [("strict", E.test_instruction_following_strict), ("loose", E.test_instruction_following_loose)]:
            outs = [fn(inp, p2r) for inp in inputs]
            ins = [ok for o in outs for ok in o.follow_instruction_list]
            r[f"prompt_{mode}"] = round(sum(o.follow_all_instructions for o in outs) / len(outs), 4)
            r[f"instruction_{mode}"] = round(sum(ins) / len(ins), 4)
            if mode == "strict":
                vec[cond] = [int(o.follow_all_instructions) for o in outs]
                grp = defaultdict(list)
                for o in outs:
                    for iid, ok in zip(o.instruction_id_list, o.follow_instruction_list):
                        grp["overlap" if iid in OVERLAP else "other"].append(int(ok))
                        grp["type:" + iid].append(int(ok))
                r["instruction_strict_overlap"] = f"{sum(grp['overlap'])}/{len(grp['overlap'])}"
                r["instruction_strict_other"] = f"{sum(grp['other'])}/{len(grp['other'])}"
                r["by_type"] = {k[5:]: f"{sum(v)}/{len(v)}" for k, v in sorted(grp.items()) if k.startswith("type:")}
        r["prompt_strict_n"] = f"{sum(vec[cond])}/{len(vec[cond])}"
        toks = sorted(g["tokens"] for g in gens)
        r["median_tokens"] = toks[len(toks) // 2]
        r["near_max_tokens"] = sum(t >= MAX_NEW - 8 for t in toks)
        res[cond] = r
    cmp = {}
    for base in ["B0", "B1"]:
        if base in vec and "M1" in vec:
            a, m = vec[base], vec["M1"]
            b = sum(1 for x, y in zip(a, m) if x and not y)
            c = sum(1 for x, y in zip(a, m) if y and not x)
            cmp[f"M1_vs_{base}"] = {"diff": round((sum(m) - sum(a)) / len(a), 4), "base_only": b, "m1_only": c,
                                    "mcnemar_p": float(f"{mcnemar_exact(b, c):.3g}")}
    out = {"max_new_tokens": MAX_NEW, "conditions": res, "comparisons": cmp}
    (ROOT / "results/ifeval/metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_type"} for k, v in res.items()} | {"cmp": cmp}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "report"])
    ap.add_argument("--cond", choices=["B0", "B1", "M1"])
    a = ap.parse_args()
    generate_cond(a.cond) if a.mode == "generate" else report()
