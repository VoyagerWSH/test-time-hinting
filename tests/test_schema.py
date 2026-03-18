from tth.schema import (
    is_correct,
    norm_answer,
    parse_base_correct,
    parse_checker_json,
    parse_experimenter_json,
    parse_hint_json,
)


def test_parse_hint_json():
    out, err = parse_hint_json('{"hint":["one","two"]}')
    assert err is None
    assert out == {"hint": ["one", "two"]}

    out, err = parse_hint_json('{"hint":["only"]}')
    assert err is None
    assert out == {"hint": ["only"]}

    out, err = parse_hint_json("not json")
    assert out is None
    assert err is not None


def test_parse_checker_json():
    out, err = parse_checker_json('{"verdict":"pass","feedback":"","hint":["a"]}')
    assert err is None
    assert out["verdict"] == "pass"
    assert out["feedback"] == ""
    assert out["hint"] == ["a"]

    out, err = parse_checker_json('{"verdict":"revise","feedback":"fix it","hint":["b"]}')
    assert err is None
    assert out["verdict"] == "revise"
    assert out["feedback"] == "fix it"


def test_parse_experimenter_json():
    out, err = parse_experimenter_json('{"answer":"A","reasoning":"Because."}')
    assert err is None
    assert out["answer"] == "A"
    assert out["reasoning"] == "Because."


def test_norm_answer():
    assert norm_answer("  A  ") == "a"
    assert norm_answer(None) is None
    assert norm_answer("") is None


def test_is_correct():
    assert is_correct("A", "A") is True
    assert is_correct("a", "A") is True
    assert is_correct("B", "A") is False


def test_parse_base_correct():
    assert parse_base_correct(True) is True
    assert parse_base_correct(False) is False
    assert parse_base_correct("true") is True
    assert parse_base_correct("false") is False
    assert parse_base_correct(1) is True
    assert parse_base_correct(0) is False


if __name__ == "__main__":
    test_parse_hint_json()
    test_parse_checker_json()
    test_parse_experimenter_json()
    test_norm_answer()
    test_is_correct()
    test_parse_base_correct()
    print("All tests passed.")
