
import math


def transition_probability(
    road_distance,
    gps_distance,
    beta=50.0,
    connected=True
):
    """
    Calculate transition probability score between
    two candidate road states.

    Parameters:
        road_distance: Shortest distance through road network (metres)
        gps_distance: Straight-line GPS displacement (metres)
        beta: Distance tolerance (metres)
        connected: Whether a valid road-network path exists

    Returns:
        Probability-like transition score between 0 and 1
    """

    if beta <= 0:
        raise ValueError("beta must be greater than zero")

    if road_distance < 0 or gps_distance < 0:
        raise ValueError("Distances cannot be negative")

    # Reject impossible transitions
    if not connected:
        return 0.0

    # Calculate distance difference
    distance_difference = abs(road_distance - gps_distance)

    # Calculate transition score
    score = math.exp(-distance_difference / beta)

    return score


def log_transition_probability(
    road_distance,
    gps_distance,
    beta=50.0,
    connected=True
):
    """
    Return log transition score for Viterbi decoding.
    """

    probability = transition_probability(
        road_distance,
        gps_distance,
        beta,
        connected
    )

    if probability == 0:
        return float("-inf")

    return math.log(probability)


if __name__ == "__main__":

    # Example 1: Good transition
    score1 = transition_probability(
        road_distance=110,
        gps_distance=100
    )

    # Example 2: Poor transition
    score2 = transition_probability(
        road_distance=300,
        gps_distance=100
    )

    # Example 3: Impossible transition
    score3 = transition_probability(
        road_distance=100,
        gps_distance=100,
        connected=False
    )

    print("Good transition:", score1)
    print("Poor transition:", score2)
    print("Impossible transition:", score3)
