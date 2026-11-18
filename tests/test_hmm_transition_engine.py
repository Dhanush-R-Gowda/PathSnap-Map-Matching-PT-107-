
"""Synthetic tests for the PathSnap HMM transition engine."""

import math

from src.utils.geometry import build_projection
from hmm.hmm_transition_engine import DirectedRoadNetwork


# Five nodes: 0 -> 1 -> 2, and a disconnected arc 3 -> 4.
nodes = [
    (0.000, 0.000),
    (0.001, 0.000),
    (0.002, 0.000),
    (0.010, 0.000),
    (0.011, 0.000),
]

arcs = [
    (0, 1),
    (1, 2),
    (3, 4),
]

track = [
    (0.000, 0.000, 0),
    (0.011, 0.000, 1),
]

proj = build_projection(track)

network = DirectedRoadNetwork(nodes, arcs, proj)


# Test 1: Connected arcs, halfway along each.
distance = network.network_distance(0, 0.5, 1, 0.5)
expected = (
    0.5 * network.arc_lengths[0]
    + 0.5 * network.arc_lengths[1]
)

assert math.isclose(distance, expected, rel_tol=1e-9)
print("PASS: Connected arc transition:", distance, "metres")


# Test 2: Forward movement on the same arc.
distance = network.network_distance(0, 0.2, 0, 0.8)
expected = 0.6 * network.arc_lengths[0]

assert math.isclose(distance, expected, rel_tol=1e-9)
print("PASS: Same-arc forward movement:", distance, "metres")


# Test 3: Disconnected road transition.
distance = network.network_distance(0, 0.5, 2, 0.5)

assert math.isinf(distance)
print("PASS: Disconnected transition rejected")


# Test 4: Backward movement without a directed loop.
distance = network.network_distance(0, 0.8, 0, 0.2)

assert math.isinf(distance)
print("PASS: Backward movement without loop rejected")


# Test 5: Log score for an impossible transition.
score = network.log_transition_score(0, 0.5, 2, 0.5, 100)

assert score == float("-inf")
print("PASS: Impossible transition has -infinity log score")

print("\nAll synthetic transition tests passed!")

