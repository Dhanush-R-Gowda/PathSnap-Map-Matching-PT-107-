
import math


def viterbi_decode(candidates, log_emissions, log_transitions):

    if not candidates or any(not states for states in candidates):
        raise ValueError("Candidate states cannot be empty")

    # Initialization
    scores = {}
    backpointers = [{}]

    for state in candidates[0]:
        scores[state] = log_emissions[0][state]

    # Dynamic programming
    for t in range(1, len(candidates)):

        new_scores = {}
        current_backpointers = {}

        for current in candidates[t]:

            best_score = float("-inf")
            best_previous = None

            for previous in candidates[t - 1]:

                transition = log_transitions.get(
                    (previous, current), float("-inf")
                )

                emission = log_emissions[t][current]

                score = (
                    scores[previous]
                    + transition
                    + emission
                )

                if score > best_score:
                    best_score = score
                    best_previous = previous

            new_scores[current] = best_score
            current_backpointers[current] = best_previous

        scores = new_scores
        backpointers.append(current_backpointers)

    # Termination
    last_state = max(scores, key=scores.get)
    best_score = scores[last_state]

    if best_score == float("-inf"):
        raise ValueError("No valid road sequence exists")

    # Backtracking
    path = [last_state]

    for t in range(len(candidates) - 1, 0, -1):

        previous = backpointers[t][path[-1]]

        if previous is None:
            raise ValueError("No valid path exists")

        path.append(previous)

    path.reverse()

    return path, best_score


if __name__ == "__main__":

    candidates = [
        ["A", "B"],
        ["B", "C"],
        ["C", "D"]
    ]

    # Example log-emission scores
    log_emissions = [
        {"A": math.log(0.8), "B": math.log(0.2)},
        {"B": math.log(0.7), "C": math.log(0.3)},
        {"C": math.log(0.6), "D": math.log(0.4)}
    ]

    # Example log-transition scores
    log_transitions = {
        ("A", "B"): math.log(0.9),
        ("A", "C"): math.log(0.1),
        ("B", "B"): math.log(0.3),
        ("B", "C"): math.log(0.8),
        ("B", "D"): math.log(0.1),
        ("C", "C"): math.log(0.4),
        ("C", "D"): math.log(0.9)
    }

    path, score = viterbi_decode(
        candidates,
        log_emissions,
        log_transitions
    )

    print("Best road sequence:", path)
    print("Best log score:", score)
