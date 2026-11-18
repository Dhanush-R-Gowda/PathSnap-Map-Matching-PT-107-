
import math


def emission_probability(distance, sigma=10.0):
    """
    Calculate Gaussian emission probability density.

    distance: GPS-to-road perpendicular distance in metres
    sigma: assumed GPS measurement error in metres
    """

    if distance < 0:
        raise ValueError("Distance cannot be negative")

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero")

    coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)

    exponent = math.exp(
        -(distance ** 2) / (2 * sigma ** 2)
    )

    return coefficient * exponent


def log_emission_probability(distance, sigma=10.0):
    """
    Calculate log Gaussian emission probability density.
    """

    if distance < 0:
        raise ValueError("Distance cannot be negative")

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero")

    return (
        -0.5 * math.log(2 * math.pi)
        - math.log(sigma)
        - (distance ** 2) / (2 * sigma ** 2)
    )


if __name__ == "__main__":

    roads = {
        "Road A": 5,
        "Road B": 15,
        "Road C": 40
    }

    for road, distance in roads.items():

        score = emission_probability(distance)

        print(road)
        print("Distance:", distance, "metres")
        print("Emission score:", score)
        print("----------------------")
