
import json
import math
import networkx as nx

from pathlib import Path


# --------------------------------------------------
# 1. LOAD SCENARIO FROM JSON
# --------------------------------------------------

def load_scenario_json(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        scenario = json.load(file)

    return scenario


# --------------------------------------------------
# 2. CHECK WHETHER A DEVICE FITS INSIDE A VEHICLE
# --------------------------------------------------

def check_device_fit(device, vehicle):
    """
    Check whether a device fits inside a vehicle.

    Sorting dimensions allows rotation of the device.
    This is a simplified geometric fit check.
    """

    device_dimensions = sorted([
        device["length"],
        device["width"],
        device["height"]
    ])

    vehicle_dimensions = sorted([
        vehicle["loading_length"],
        vehicle["loading_width"],
        vehicle["loading_height"]
    ])

    return all(
        device_dimensions[i] <= vehicle_dimensions[i]
        for i in range(3)
    )


# --------------------------------------------------
# 3. CALCULATE DISTANCE BETWEEN TWO LOCATIONS
# --------------------------------------------------

def calculate_distance(location_a, location_b):

    dx = location_a["x"] - location_b["x"]
    dy = location_a["y"] - location_b["y"]
    dz = location_a["z"] - location_b["z"]

    return math.sqrt(dx**2 + dy**2 + dz**2)


# --------------------------------------------------
# 4. BUILD HETEROGENEOUS GRAPH
# --------------------------------------------------

def build_graph(scenario):

    G = nx.MultiDiGraph()

    # --------------------------------------------------
    # STEP A: ADD INCIDENT NODES
    # --------------------------------------------------

    for incident in scenario["incidents"]:

        incident_id = incident["incident_id"]

        G.add_node(
            incident_id,
            node_type="incident",
            x=incident["x"],
            y=incident["y"],
            z=incident["z"],
            incident_category=incident.get(
                "incident_category", "unknown"
            ),
            severity=incident.get("severity", 0),
            required_devices=incident["required_devices"]
        )

    # --------------------------------------------------
    # STEP B: ADD DEVICE NODES
    # --------------------------------------------------

    for device in scenario["devices"]:

        device_id = device["device_id"]

        G.add_node(
            device_id,
            node_type="device",
            device_type=device["device_type"],
            operating_cost=device["operating_cost"],
            length=device["length"],
            width=device["width"],
            height=device["height"]
        )

    # --------------------------------------------------
    # STEP C: ADD STORAGE NODES
    # --------------------------------------------------

    for storage in scenario["storages"]:

        storage_id = storage["storage_id"]

        G.add_node(
            storage_id,
            node_type="storage",
            x=storage["x"],
            y=storage["y"],
            z=storage["z"]
        )

    # --------------------------------------------------
    # STEP D: ADD VEHICLE NODES
    # --------------------------------------------------

    for vehicle in scenario["vehicles"]:

        vehicle_id = vehicle["vehicle_id"]

        G.add_node(
            vehicle_id,
            node_type="vehicle",
            x=vehicle["x"],
            y=vehicle["y"],
            z=vehicle["z"],
            loading_length=vehicle["loading_length"],
            loading_width=vehicle["loading_width"],
            loading_height=vehicle["loading_height"]
        )

    # --------------------------------------------------
    # STEP E: STORAGE -> DEVICE
    # Relationship: stores
    # --------------------------------------------------

    for storage in scenario["storages"]:

        storage_id = storage["storage_id"]

        for device_id in storage["device_ids"]:

            if not G.has_node(device_id):
                raise ValueError(
                    f"Unknown device {device_id} "
                    f"in storage {storage_id}"
                )

            G.add_edge(
                storage_id,
                device_id,
                relation="stores"
            )

    # --------------------------------------------------
    # STEP F: INCIDENT -> DEVICE
    # Relationship: requires
    # --------------------------------------------------

    for incident in scenario["incidents"]:

        incident_id = incident["incident_id"]

        requirements = incident["required_devices"]

        for device in scenario["devices"]:

            device_id = device["device_id"]
            device_type = device["device_type"]

            if device_type in requirements:

                G.add_edge(
                    incident_id,
                    device_id,
                    relation="requires",
                    quantity=requirements[device_type],
                    required_device_type=device_type
                )

    # --------------------------------------------------
    # STEP G: DEVICE -> VEHICLE
    # Relationship: can_be_transported_by
    # --------------------------------------------------

    for device in scenario["devices"]:

        device_id = device["device_id"]

        for vehicle in scenario["vehicles"]:

            vehicle_id = vehicle["vehicle_id"]

            fits = check_device_fit(device, vehicle)

            if fits:

                G.add_edge(
                    device_id,
                    vehicle_id,
                    relation="can_be_transported_by",
                    fits=True
                )

    # --------------------------------------------------
    # STEP H: VEHICLE -> INCIDENT
    # Relationship: can_serve
    # --------------------------------------------------

    for vehicle in scenario["vehicles"]:

        vehicle_id = vehicle["vehicle_id"]

        for incident in scenario["incidents"]:

            incident_id = incident["incident_id"]

            distance = calculate_distance(
                vehicle,
                incident
            )

            G.add_edge(
                vehicle_id,
                incident_id,
                relation="can_serve",
                distance=round(distance, 2)
            )

    return G


# --------------------------------------------------
# 5. BUILD GRAPH DIRECTLY FROM JSON FILE
# --------------------------------------------------

def build_graph_from_json(file_path):

    scenario = load_scenario_json(file_path)

    return build_graph(scenario)


# --------------------------------------------------
# 6. DISPLAY GRAPH INFORMATION
# --------------------------------------------------

def print_graph_summary(G):

    print("\n" + "=" * 55)
    print("DYNAMIC HETEROGENEOUS GRAPH")
    print("=" * 55)

    print("\nTOTAL GRAPH SIZE")
    print("-" * 30)

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    print("\nNODE COUNTS BY TYPE")
    print("-" * 30)

    node_types = {}

    for _, attributes in G.nodes(data=True):

        node_type = attributes["node_type"]

        node_types[node_type] = (
            node_types.get(node_type, 0) + 1
        )

    for node_type, count in node_types.items():
        print(f"{node_type}: {count}")

    print("\nEDGE COUNTS BY RELATIONSHIP")
    print("-" * 30)

    relationship_counts = {}

    for _, _, attributes in G.edges(data=True):

        relation = attributes["relation"]

        relationship_counts[relation] = (
            relationship_counts.get(relation, 0) + 1
        )

    for relation, count in relationship_counts.items():
        print(f"{relation}: {count}")

    print("\nSAMPLE NODES")
    print("-" * 30)

    for node, attributes in list(G.nodes(data=True))[:5]:
        print(node, attributes)

    print("\nSAMPLE EDGES")
    print("-" * 30)

    for source, target, attributes in list(
        G.edges(data=True)
    )[:10]:

        print(
            source,
            "->",
            target,
            attributes
        )

    print("\n" + "=" * 55)


# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    scenario_file = "data/sample/Scenario_001.json"

    graph = build_graph_from_json(scenario_file)

    print_graph_summary(graph)
