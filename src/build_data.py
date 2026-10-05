"""학습·시험·대조 문항을 만든다(docs/blueprint.md 3절). 분할 사이 주제·값·문항이 겹치면 멈춘다.

    python src/build_data.py   ->  data/train.jsonl, data/test.jsonl, data/control_free.jsonl, data/control_math.jsonl
"""
import json
import random
from pathlib import Path

from constraints import CONSTRAINTS, END_PHRASES_TEST, END_PHRASES_TRAIN, compatible, instruction

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234

TOPICS = [
    "learning a new language", "saving money", "growing tomatoes", "remote work", "public transport", "drinking water",
    "reading habits", "morning routines", "recycling", "learning to swim", "baking bread", "city cycling",
    "time management", "volunteering", "camping", "studying for exams", "making friends", "keeping a journal",
    "home workouts", "cooking rice", "visiting museums", "learning guitar", "caring for a cat", "caring for a dog",
    "planting trees", "using a library", "writing emails", "team meetings", "online shopping", "rainy days",
    "photography", "chess", "board games", "hiking", "first aid", "healthy sleep", "stretching", "learning to code",
    "budget travel", "minimalism", "coffee", "tea", "bird watching", "painting", "knitting", "running", "yoga",
    "public speaking", "job interviews", "moving house", "cleaning a kitchen", "fixing a bike", "night sky",
    "ocean tides", "volcanoes", "honey bees", "recipes for soup", "family dinners", "learning history", "solar panels",
    # 시험 주제(뒤 20개)
    "train journeys", "making pizza", "lighthouses", "street markets", "winter clothing", "paper airplanes",
    "desert plants", "learning to draw", "kite flying", "local festivals", "ice skating", "postcards", "rock climbing",
    "making pasta", "sunflowers", "fishing", "puzzles", "picnics", "old maps", "clouds",
]
TRAIN_TOPICS, TEST_TOPICS = TOPICS[:60], TOPICS[60:]
TASKS = ["Write a short paragraph about {t}.", "Give a beginner some advice about {t}.", "Explain why people enjoy {t}."]

VALUES = {
    "train": {"bullets": [3, 4, 5], "max_words": [40, 50, 60], "keyword": ["simple", "daily", "practice", "small"],
              "keys": [("title", "summary"), ("topic", "tip")], "phrase": END_PHRASES_TRAIN},
    "test": {"bullets": [2, 6], "max_words": [30, 70], "keyword": ["often", "careful"],
             "keys": [("name", "reason")], "phrase": END_PHRASES_TEST},
}


def _args(name, split, rng):
    v = VALUES[split]
    if name == "bullets":
        return {"n": rng.choice(v["bullets"])}
    if name == "max_words":
        return {"n": rng.choice(v["max_words"])}
    if name == "keyword_twice":
        return {"word": rng.choice(v["keyword"])}
    if name == "json_keys":
        return {"keys": list(rng.choice(v["keys"]))}
    if name == "end_phrase":
        return {"phrase": rng.choice(v["phrase"])}
    return {}


def _item(topic, split, rng, idx):
    names = list(CONSTRAINTS)
    while True:
        k = rng.choice([1, 2])
        picked = rng.sample(names, k)
        if compatible(picked):
            break
    item = {"id": f"{split}-{idx:04d}", "topic": topic,
            "constraints": [{"name": n, "args": _args(n, split, rng)} for n in picked]}
    item["prompt"] = rng.choice(TASKS).format(t=topic) + " " + instruction(item)
    return item


def build():
    rng = random.Random(SEED)
    train = [_item(t, "train", rng, i * 6 + j) for i, t in enumerate(TRAIN_TOPICS) for j in range(6)]
    test = [_item(t, "test", rng, i * 8 + j) for i, t in enumerate(TEST_TOPICS) for j in range(8)]
    free = [{"id": f"free-{i:03d}", "topic": t, "prompt": rng.choice(TASKS).format(t=t), "constraints": []}
            for i, t in enumerate(TEST_TOPICS * 2)]
    math = []
    for i in range(60):
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        math.append({"id": f"math-{i:03d}", "prompt": f"What is {a} {op} {b}? Answer with just the number.", "answer": ans})
    _assert_disjoint(train, test)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    for name, rows in [("train", train), ("test", test), ("control_free", free), ("control_math", math)]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    print(f"train {len(train)} · test {len(test)} · control_free {len(free)} · control_math {len(math)}")


def _assert_disjoint(train, test):
    assert not set(TRAIN_TOPICS) & set(TEST_TOPICS), "주제 겹침"
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in test}, "문항 겹침"
    for key in ("bullets", "max_words", "keyword", "phrase"):
        assert not set(VALUES["train"][key]) & set(VALUES["test"][key]), f"값 겹침: {key}"
    assert not {k for p in VALUES["train"]["keys"] for k in p} & {k for p in VALUES["test"]["keys"] for k in p}, "JSON 키 겹침"


if __name__ == "__main__":
    build()
