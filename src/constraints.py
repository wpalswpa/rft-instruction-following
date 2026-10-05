"""검사 가능한 형식 제약 8종: 지시문 생성과 결정적 판정(docs/blueprint.md 2절)."""
import json
import re

END_PHRASES_TRAIN = ["Hope this helps.", "That is all for now.", "Thanks for reading.", "Stay curious."]
END_PHRASES_TEST = ["Any other questions?", "Let me know what you think."]


def _strip(text):
    return text.strip()


def check_json_keys(text, keys):
    try:
        obj = json.loads(_strip(text))
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and set(obj.keys()) == set(keys)


def check_bullets(text, n):
    lines = [ln.strip() for ln in _strip(text).splitlines() if ln.strip()]
    return len(lines) == n and all(ln.startswith("- ") for ln in lines)


def check_lowercase(text):
    t = _strip(text)
    return bool(re.search(r"[A-Za-z]", t)) and t == t.lower()


def check_max_words(text, n):
    return 0 < len(_strip(text).split()) <= n


def check_keyword_twice(text, word):
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)) >= 2


def check_end_phrase(text, phrase):
    return _strip(text).endswith(phrase)


def check_no_commas(text):
    t = _strip(text)
    return bool(t) and "," not in t


def check_quote_wrap(text):
    t = _strip(text)
    return len(t) >= 2 and t.startswith('"') and t.endswith('"')


# 이름 -> (지시문 함수, 판정 함수). 지시문과 판정은 같은 인자(dict)를 받는다.
CONSTRAINTS = {
    "json_keys": (lambda a: f"Respond with only a JSON object (no code block) that has exactly the keys \"{a['keys'][0]}\" and \"{a['keys'][1]}\".",
                  lambda t, a: check_json_keys(t, a["keys"])),
    "bullets": (lambda a: f"Answer with exactly {a['n']} bullet points, each line starting with \"- \", and nothing else.",
                lambda t, a: check_bullets(t, a["n"])),
    "lowercase": (lambda a: "Your entire response must be in lowercase letters only.",
                  lambda t, a: check_lowercase(t)),
    "max_words": (lambda a: f"Answer in at most {a['n']} words.",
                  lambda t, a: check_max_words(t, a["n"])),
    "keyword_twice": (lambda a: f"Use the word \"{a['word']}\" at least twice.",
                      lambda t, a: check_keyword_twice(t, a["word"])),
    "end_phrase": (lambda a: f"End your response with the exact phrase \"{a['phrase']}\"",
                   lambda t, a: check_end_phrase(t, a["phrase"])),
    "no_commas": (lambda a: "Do not use any commas in your response.",
                  lambda t, a: check_no_commas(t)),
    "quote_wrap": (lambda a: "Wrap your entire response in double quotation marks.",
                   lambda t, a: check_quote_wrap(t)),
}

# 서로 모순되거나 판정이 겹치는 조합을 막는다. 여기 없는 쌍만 함께 쓴다.
INCOMPATIBLE = {
    frozenset(p) for p in [
        ("json_keys", "bullets"), ("json_keys", "end_phrase"), ("json_keys", "quote_wrap"), ("json_keys", "lowercase"),
        ("json_keys", "max_words"), ("json_keys", "keyword_twice"), ("json_keys", "no_commas"),
        ("bullets", "quote_wrap"), ("bullets", "end_phrase"), ("quote_wrap", "end_phrase"),
        ("max_words", "bullets"),
    ]
}


def compatible(names):
    return all(frozenset((a, b)) not in INCOMPATIBLE for i, a in enumerate(names) for b in names[i + 1:])


def instruction(item):
    return " ".join(CONSTRAINTS[c["name"]][0](c["args"]) for c in item["constraints"])


def judge(text, item):
    """문항의 제약마다 통과 여부. 반환: {이름: bool}."""
    return {c["name"]: bool(CONSTRAINTS[c["name"]][1](text, c["args"])) for c in item["constraints"]}
