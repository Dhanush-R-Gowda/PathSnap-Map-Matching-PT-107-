
"""Log-space Viterbi decoder for PathSnap."""

import math


def _validate_log_score(value, description):
    """Validate a log probability or log score."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be numeric")

    if math.isnan(value) or value == float("inf"):
        raise ValueError(
            f"{description} cannot be NaN or positive infinity"
        )


def viterbi_decode(candidates, log_emissions, log_transitions):
    """Find the most likely sequence of candidate road states."""

    if not candidates:
        raise ValueError("Candidate observations cannot be empty")

    if any(not states for states in candidates):
        raise ValueError("At least one observation has no candidates")

    if len(log_emissions) != len(candidates):
        raise ValueError("Emission count must match observation count")

    if not isinstance(log_transitions, dict):
        raise TypeError("log_transitions must be a dictionary")

    # Validate emissions
    for t, states in enumerate(candidates):
        if len(states) != len(set(states)):
            raise ValueError(f"Duplicate candidates at observation {t}")

        for state in states:
            if state not in log_emissions[t]:
                raise ValueError(
                    f"Missing emission score for {state} at observation {t}"
                )

            _validate_log_score(
                log_emissions[t][state],
                f"Emission score for {state}"
            )

    # Initialization
    scores = {
        state: float(log_emissions[0][state])
        for state in candidates[0]
    }

    backpointers = [{}]

    # Dynamic programming
    for t in range(1, len(candidates)):
        new_scores = {}
        current_backpointers = {}

        for current in candidates[t]:
            best_score = float("-inf")
            best_previous = None

            emission = float(log_emissions[t][current])

            for previous in candidates[t - 1]:
                transition = log_transitions.get(
                    (previous, current),
                    float("-inf")
                )

                _validate_log_score(
                    transition,
                    f"Transition {previous} -> {current}"
                )

                if scores[previous] == float("-inf"):
                    continue

                if transition == float("-inf"):
                    continue

                score = scores[previous] + transition + emission

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
