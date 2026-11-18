
import networkx as nx

# Create a road network graph
G = nx.Graph()

# Add road intersections and distances (metres)
G.add_weighted_edges_from([
    ("A", "B", 100),
    ("B", "C", 150),
    ("C", "D", 200),
    ("A", "D", 600),
    ("B", "D", 250)
])

# Calculate shortest road distance
distance = nx.shortest_path_length(
    G,
    source="A",
    target="D",
    weight="weight"
)

# Find the actual shortest route
path = nx.shortest_path(
    G,
    source="A",
    target="D",
    weight="weight"
)

print("Shortest road path:", path)
print("Shortest road distance:", distance, "metres")
