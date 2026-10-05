"""세 조건(B0 기본·B1 기본+엄격 시스템 문구·M1 미세조정)을 같은 시험 문항으로 평가한다(blueprint 4.3, 5절).

    python src/evaluate.py generate --cond B0|B1|M1   ->  results/gen_<cond>.jsonl (시험·대조 문항 생성 결과)
    python src/evaluate.py report                     ->  results/metrics.json
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from math import comb
from pathlib import Path

from constraints import END_PHRASES_TEST, END_PHRASES_TRAIN, check_lowercase, check_quote_wrap, judge

ROOT = Path(__file__).resolve().parents[1]
SETS = ["test", "control_free", "control_math"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def generate_cond(cond):
    from lm import generate, load
    adapter = ROOT / "outputs/lora" if (ROOT / "outputs/lora").exists() else ROOT / "adapter"  # 학습 직후 산출물, 없으면 저장소의 어댑터
    tok, model = load(adapter_dir=adapter if cond == "M1" else None)
    rows, t0 = [], time.time()
    for name in SETS:
        items = load_jsonl(ROOT / f"data/{name}.jsonl")
        outs = generate(tok, model, [it["prompt"] for it in items], strict=(cond == "B1"),
                        on_batch=lambda d, n: print(f"\r{cond} {name} {d}/{n} {time.time() - t0:.0f}s", end="", flush=True))
        print()
        rows += [{"set": name, "id": it["id"], "response": o[0]} for it, o in zip(items, outs)]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/gen_{cond}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def spurious(text):
    """요청하지 않은 형식: 전부 소문자·큰따옴표 감싸기·정해 둔 끝 문구·글머리표로만 된 답."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    bullets_only = len(lines) >= 2 and all(l.startswith("- ") for l in lines)
    ends = any(text.strip().endswith(p) for p in END_PHRASES_TRAIN + END_PHRASES_TEST)
    return check_lowercase(text) or check_quote_wrap(text) or ends or bullets_only


def mcnemar_exact(b, c):
    """b: 기준만 통과, c: 비교 대상만 통과. 양측 정확 검정."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_diff(a, b, reps=10000, seed=7):
    rng = random.Random(seed)
    n, diffs = len(a), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return [round(diffs[int(0.025 * reps)], 4), round(diffs[int(0.975 * reps)], 4)]


def report():
    test = {it["id"]: it for it in load_jsonl(ROOT / "data/test.jsonl")}
    math_ans = {it["id"]: it["answer"] for it in load_jsonl(ROOT / "data/control_math.jsonl")}
    conds = [c for c in ["B0", "B1", "M1"] if (ROOT / f"results/gen_{c}.jsonl").exists()]
    res, strict_vec = {}, {}
    for cond in conds:
        gens = load_jsonl(ROOT / f"results/gen_{cond}.jsonl")
        by_set = defaultdict(dict)
        for g in gens:
            by_set[g["set"]][g["id"]] = g["response"]
        ids = sorted(test)
        verdicts = {i: judge(by_set["test"][i], test[i]) for i in ids}
        strict = [int(all(verdicts[i].values())) for i in ids]
        per = defaultdict(list)
        for i in ids:
            for name, ok in verdicts[i].items():
                per[name].append(int(ok))
        words = [len(by_set["test"][i].split()) for i in ids]
        math_ok = []
        for i, ans in math_ans.items():
            m = re.search(r"-?\d+", by_set["control_math"][i].replace(",", ""))
            math_ok.append(int(bool(m) and int(m.group()) == ans))
        free = list(by_set["control_free"].values())
        strict_vec[cond] = strict
        res[cond] = {
            "prompt_strict": round(sum(strict) / len(strict), 4), "prompt_strict_n": f"{sum(strict)}/{len(strict)}",
            "instruction_level": round(sum(sum(v) for v in per.values()) / sum(len(v) for v in per.values()), 4),
            "per_constraint": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per.items())},
            "median_words": sorted(words)[len(words) // 2],
            "math_accuracy": f"{sum(math_ok)}/{len(math_ok)}",
            "free_spurious": f"{sum(spurious(t) for t in free)}/{len(free)}",
        }
    comparisons = {}
    for base in ["B0", "B1"]:
        if base in strict_vec and "M1" in strict_vec:
            a, m = strict_vec[base], strict_vec["M1"]
            b = sum(1 for x, y in zip(a, m) if x and not y)
            c = sum(1 for x, y in zip(a, m) if y and not x)
            comparisons[f"M1_vs_{base}"] = {"diff": round((sum(m) - sum(a)) / len(a), 4), "base_only": b, "m1_only": c,
                                            "mcnemar_p": float(f"{mcnemar_exact(b, c):.3g}"), "bootstrap95": bootstrap_diff(a, m)}
    verdict = None
    if "M1_vs_B1" in comparisons and "B0" in res:
        cmp = comparisons["M1_vs_B1"]
        frac = lambda s: int(s.split("/")[0]) / int(s.split("/")[1])
        c1 = cmp["diff"] >= 0.10 and cmp["mcnemar_p"] < 0.05
        c2 = frac(res["B0"]["math_accuracy"]) - frac(res["M1"]["math_accuracy"]) <= 0.05
        c3 = frac(res["M1"]["free_spurious"]) <= 0.10
        verdict = {"c1_beats_prompt_by_10pp": c1, "c2_math_drop_le_5pp": c2, "c3_spurious_le_10pct": c3,
                   "label": "이 조건에서 효과 있음" if c1 and c2 and c3 else ("부작용 있음" if not (c2 and c3) else "프롬프트 대비 이점 확인 못 함")}
    out = {"conditions": res, "comparisons": comparisons, "verdict": verdict}
    (ROOT / "results/metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "report"])
    ap.add_argument("--cond", choices=["B0", "B1", "M1"])
    a = ap.parse_args()
    generate_cond(a.cond) if a.mode == "generate" else report()
