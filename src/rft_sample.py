"""거부 샘플링: 학습 문항마다 기본 모델 응답 k개를 뽑아 모든 제약을 통과한 것 중 가장 짧은 것 하나를 남긴다(blueprint 4.1).

    python src/rft_sample.py [--k 4]  ->  data/rft_samples.jsonl(전체 표본), data/rft_train.jsonl(학습용)
"""
import argparse
import json
import time
from pathlib import Path

from constraints import judge
from lm import generate, load

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()
    items = [json.loads(l) for l in (ROOT / "data/train.jsonl").read_text("utf-8").splitlines()]
    tok, model = load()
    t0 = time.time()
    outs = generate(tok, model, [it["prompt"] for it in items], sample=True, n=args.k, seed=2026, max_new_tokens=args.max_new_tokens,
                    on_batch=lambda d, n: print(f"\r{d}/{n} {time.time() - t0:.0f}s", end="", flush=True))
    print()
    samples, train = [], []
    for it, responses in zip(items, outs):
        verdicts = [judge(r, it) for r in responses]
        passed = [r for r, v in zip(responses, verdicts) if all(v.values())]
        samples.append({"id": it["id"], "responses": responses, "verdicts": verdicts})
        if passed:
            train.append({"id": it["id"], "prompt": it["prompt"], "response": min(passed, key=len)})
    (ROOT / "data/rft_samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), "utf-8")
    (ROOT / "data/rft_train.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in train), "utf-8")
    n_pass = sum(all(v.values()) for s in samples for v in s["verdicts"])
    print(f"표본 {len(items) * args.k}개 중 통과 {n_pass}개, 학습 문항 {len(train)}/{len(items)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
