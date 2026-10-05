"""판정 함수가 의도한 행동을 하는지: 통과 예와 실패 예를 함께 둔다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constraints import (check_bullets, check_end_phrase, check_json_keys, check_keyword_twice, check_lowercase,  # noqa: E402
                         check_max_words, check_no_commas, check_quote_wrap, compatible, judge)


def test_json_keys_exact_and_no_code_block():
    assert check_json_keys('{"title": "a", "summary": "b"}', ["title", "summary"])
    assert not check_json_keys('```json\n{"title": "a", "summary": "b"}\n```', ["title", "summary"])
    assert not check_json_keys('{"title": "a"}', ["title", "summary"])
    assert not check_json_keys('{"title": "a", "summary": "b", "x": 1}', ["title", "summary"])
    assert not check_json_keys('["title", "summary"]', ["title", "summary"])


def test_bullets_count_and_no_extra_lines():
    assert check_bullets("- a\n- b\n- c", 3)
    assert check_bullets("- a\n\n- b\n- c\n", 3)
    assert not check_bullets("Here you go:\n- a\n- b\n- c", 3)
    assert not check_bullets("- a\n- b", 3)
    assert not check_bullets("* a\n* b\n* c", 3)


def test_lowercase_requires_letters():
    assert check_lowercase("all lower case here.")
    assert not check_lowercase("Capital first.")
    assert not check_lowercase("123 456")


def test_max_words_counts_whitespace_tokens():
    assert check_max_words("one two three", 3)
    assert not check_max_words("one two three four", 3)
    assert not check_max_words("   ", 3)


def test_keyword_twice_word_boundary_case_insensitive():
    assert check_keyword_twice("River runs. The river bends.", "river")
    assert not check_keyword_twice("rivers and riverbank", "river")
    assert not check_keyword_twice("one river only", "river")


def test_end_phrase_after_trailing_space():
    assert check_end_phrase("Text. Hope this helps.  \n", "Hope this helps.")
    assert not check_end_phrase("Hope this helps. More text.", "Hope this helps.")


def test_no_commas_and_quote_wrap():
    assert check_no_commas("no commas here.")
    assert not check_no_commas("a, b")
    assert check_quote_wrap('"wrapped answer"')
    assert not check_quote_wrap('"only start')
    assert not check_quote_wrap('"')


def test_incompatible_pairs_are_rejected():
    assert not compatible(["json_keys", "bullets"])
    assert not compatible(["bullets", "quote_wrap"])
    assert compatible(["lowercase", "no_commas"])


def test_judge_reports_each_constraint():
    item = {"constraints": [{"name": "lowercase", "args": {}}, {"name": "no_commas", "args": {}}]}
    assert judge("all good here", item) == {"lowercase": True, "no_commas": True}
    assert judge("All good, here", item) == {"lowercase": False, "no_commas": False}
