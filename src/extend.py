"""확장 실험 E1: 시드 반복과 선택 규칙 비교(blueprint 10.1).

    python src/extend.py select --rule S|R --seed 43   ->  data/ext/train_<rule>_s<seed>.jsonl
    python src/extend.py report                         ->  results/ext/metrics.json

학습과 생성은 train_lora.py·evaluate.py를 그대로 쓴다(run_ext.sh). S·42는 첫 실험의 M1(results/gen_M1.jsonl)이다.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from constraints import judge
from evaluate import load_jsonl, spurious
import re

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("S", 42), ("S", 43), ("S", 44), ("R", 42), ("R", 43), ("R", 44)]


def select(rule, seed):
    items = {it["id"]: it for it in load_jsonl(ROOT / "data/train.jsonl")}
    rng = random.Random(seed)
    rows = []
    for s in load_jsonl(ROOT / "data/rft_samples.jsonl"):
        passed = [r for r, v in zip(s["responses"], s["verdicts"]) if all(v.values())]
        if not passed:
            continue
        pick = min(passed, key=len) if rule == "S" else rng.choice(passed)
        rows.append({"id": s["id"], "prompt": items[s["id"]]["prompt"], "response": pick})
    if rule == "S":  # 첫 실험의 학습 데이터와 같아야 한다
        assert rows == load_jsonl(ROOT / "data/rft_train.jsonl"), "S 규칙이 첫 실험의 학습 데이터를 재현하지 못함"
    out = ROOT / f"data/ext/train_{rule}_s{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    words = sorted(len(r["response"].split()) for r in rows)
    print(f"{rule} s{seed}: 학습 예시 {len(rows)}개, 응답 단어 중앙값 {words[len(words) // 2]}")


def gen_path(rule, seed):
    return ROOT / ("results/gen_M1.jsonl" if (rule, seed) == ("S", 42) else f"results/ext/gen_{rule}_s{seed}.jsonl")


def score(path):
    test = {it["id"]: it for it in load_jsonl(ROOT / "data/test.jsonl")}
    math_ans = {it["id"]: it["answer"] for it in load_jsonl(ROOT / "data/control_math.jsonl")}
    by_set = defaultdict(dict)
    for g in load_jsonl(path):
        by_set[g["set"]][g["id"]] = g["response"]
    per = defaultdict(list)
    strict = []
    for i in sorted(test):
        v = judge(by_set["test"][i], test[i])
        strict.append(int(all(v.values())))
        for k, ok in v.items():
            per[k].append(int(ok))
    words = sorted(len(by_set["test"][i].split()) for i in test)
    math_ok = 0
    for i, ans in math_ans.items():
        m = re.search(r"-?\d+", by_set["control_math"][i].replace(",", ""))
        math_ok += int(bool(m) and int(m.group()) == ans)
    return {"strict": sum(strict), "n": len(strict), "median_words": words[len(words) // 2],
            "per_constraint": {k: sum(v) for k, v in sorted(per.items())},
            "math": math_ok, "spurious": sum(spurious(t) for t in by_set["control_free"].values())}


def report():
    runs = {f"{r}_s{s}": score(gen_path(r, s)) for r, s in RUNS if gen_path(r, s).exists()}
    b1 = score(ROOT / "results/gen_B1.jsonl")
    summary = {}
    for rule in ["S", "R"]:
        xs = [v for k, v in runs.items() if k.startswith(rule + "_")]
        if not xs:
            continue
        rates = [x["strict"] / x["n"] for x in xs]
        summary[rule] = {"runs": len(xs), "strict_counts": [x["strict"] for x in xs],
                         "strict_mean": round(statistics.mean(rates), 4),
                         "strict_sd": round(statistics.stdev(rates), 4) if len(xs) > 1 else None,
                         "median_words": [x["median_words"] for x in xs],
                         "max_words_pass": [x["per_constraint"].get("max_words") for x in xs],
                         "math": [x["math"] for x in xs], "spurious": [x["spurious"] for x in xs]}
    b1_rate = b1["strict"] / b1["n"]
    judgement = {}
    if "S" in summary and summary["S"]["runs"] == 3:
        judgement["a_seed_stable"] = all(c >= 39 for c in summary["S"]["strict_counts"])
    if "R" in summary and summary["R"]["runs"] == 3:
        judgement["b_R_beats_B1_by_10pp"] = summary["R"]["strict_mean"] - b1_rate >= 0.10
    out = {"B1": b1, "runs": runs, "summary": summary, "judgement_blueprint_10_1": judgement}
    (ROOT / "results/ext").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/ext/metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"summary": summary, "judgement": judgement}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "report"])
    ap.add_argument("--rule", choices=["S", "R"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    select(a.rule, a.seed) if a.mode == "select" else report()
