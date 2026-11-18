
import math
import pytest

from hmm.viterbi_decoder import viterbi_decode


def test_valid_path():
    candidates = [["A", "B"], ["C", "D"]]

    emissions = [
        {"A": math.log(0.8), "B": math.log(0.2)},
        {"C": math.log(0.7), "D": math.log(0.3)}
    ]

    transitions = {
        ("A", "C"): math.log(0.9),
        ("A", "D"): math.log(0.1),
        ("B", "C"): math.log(0.2),
        ("B", "D"): math.log(0.8)
    }

    path, score = viterbi_decode(
        candidates, emissions, transitions
    )

    assert path == ["A", "C"]
    assert math.isclose(
        score,
        math.log(0.8) + math.log(0.9) + math.log(0.7)
    )


def test_impossible_path():
    candidates = [["A"], ["B"]]

    emissions = [
        {"A": 0.0},
        {"B": 0.0}
    ]

    transitions = {}

    with pytest.raises(ValueError, match="No valid road sequence"):
        viterbi_decode(candidates, emissions, transitions)


def test_empty_observation():
    with pytest.raises(ValueError):
        viterbi_decode(
            [["A"], []],
            [{"A": 0.0}, {}],
            {}
        )


def test_missing_emission():
    with pytest.raises(ValueError, match="Missing emission"):
        viterbi_decode(
            [["A"]],
            [{}],
            {}
        )


def test_missing_transition_is_impossible():
    candidates = [["A"], ["B"]]

    emissions = [
        {"A": 0.0},
        {"B": 0.0}
    ]

    with pytest.raises(ValueError):
        viterbi_decode(candidates, emissions, {})


if __name__ == "__main__":
    test_valid_path()
    test_impossible_path()
    test_empty_observation()
    test_missing_emission()
    test_missing_transition_is_impossible()

    print("All Viterbi tests passed!")
