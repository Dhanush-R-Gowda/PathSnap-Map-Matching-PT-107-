
import networkx as nx

from transition import transition_probability


# 1. Create sample road network
G = nx.Graph()

G.add_weighted_edges_from([
    ("A", "B", 100),
    ("B", "C", 150),
    ("C", "D", 200),
    ("A", "D", 600),
    ("B", "D", 250)
])


# 2. Define previous and current road positions
previous_road = "A"
current_road = "D"

# GPS displacement in metres
gps_distance = 300


# 3. Calculate shortest road distance
try:
    road_distance = nx.shortest_path_length(
        G,
        source=previous_road,
        target=current_road,
        weight="weight"
    )

    # 4. Calculate transition score
    score = transition_probability(
        road_distance=road_distance,
        gps_distance=gps_distance,
        beta=50.0,
        connected=True
    )

    # 5. Display results
    print("Previous road:", previous_road)
    print("Current road:", current_road)
    print("Shortest road distance:", road_distance, "metres")
    print("GPS displacement:", gps_distance, "metres")
    print("Transition score:", score)

except nx.NetworkXNoPath:
    print("No valid road path exists")

except nx.NodeNotFound:
    print("Road location not found in network")
