
def reconstruct_route(selected_segments, segment_geometries):
    """
    Convert selected road segment IDs into ordered coordinates.

    selected_segments:
        Ordered list of road segment IDs.

    segment_geometries:
        Dictionary mapping segment IDs to coordinate lists.
        Coordinates use (longitude, latitude).

    Returns:
        Ordered route coordinates.
    """

    if not selected_segments:
        raise ValueError("Selected road segments cannot be empty")

    route_coordinates = []

    for segment_id in selected_segments:

        if segment_id not in segment_geometries:
            raise ValueError(
                f"Missing geometry for segment: {segment_id}"
            )

        coordinates = segment_geometries[segment_id]

        if not coordinates:
            raise ValueError(
                f"Empty geometry for segment: {segment_id}"
            )

        if route_coordinates:
            # Check whether consecutive segments connect
            if route_coordinates[-1] != coordinates[0]:
                raise ValueError(
                    f"Disconnected segments near {segment_id}"
                )

            # Avoid duplicate connecting coordinate
            route_coordinates.extend(coordinates[1:])

        else:
            route_coordinates.extend(coordinates)

    return route_coordinates


if __name__ == "__main__":

    # Synthetic road geometries for testing
    segment_geometries = {

        "A": [
            (77.5900, 12.9700),
            (77.5910, 12.9710)
        ],

        "B": [
            (77.5910, 12.9710),
            (77.5920, 12.9720)
        ],

        "C": [
            (77.5920, 12.9720),
            (77.5930, 12.9730)
        ]
    }

    # Output from Viterbi (example)
    selected_segments = ["A", "B", "C"]

    route = reconstruct_route(
        selected_segments,
        segment_geometries
    )

    print("Reconstructed route coordinates:")

    for coordinate in route:
        print(coordinate)

    print("Total coordinates:", len(route))
