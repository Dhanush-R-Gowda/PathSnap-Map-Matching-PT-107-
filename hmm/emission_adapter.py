
import math


def normalize_candidate_probabilities(probabilities, epsilon=1e-12):
    """
    Normalize ML probabilities for candidates belonging
    to one GPS observation, then convert them to log scores.
    """

    if not probabilities:
        raise ValueError("Candidate probabilities cannot be empty")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
        raise ValueError("Probabilities must be finite and non-negative")

    adjusted = {
        segment_id: max(probability, epsilon)
        for segment_id, probability in probabilities.items()
    }

    total = sum(adjusted.values())

    normalized = {
        segment_id: probability / total
        for segment_id, probability in adjusted.items()
    }

    log_emissions = {
        segment_id: math.log(probability)
        for segment_id, probability in normalized.items()
    }

    return normalized, log_emissions


if __name__ == "__main__":
    ml_predictions = {
        "road_A": 0.70,
        "road_B": 0.20,
        "road_C": 0.10
    }

    probabilities, log_scores = normalize_candidate_probabilities(
        ml_predictions
    )

    print("Normalized probabilities:", probabilities)
    print("Log emission scores:", log_scores)
