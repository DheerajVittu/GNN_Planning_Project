
import torch
import networkx as nx


# --------------------------------------------------
# 1. DEVICE TYPES
# --------------------------------------------------

# Fixed vocabulary: every device type gets one numerical feature.
DEVICE_TYPES = [
    "Pump",
    "Radio",
    "MedicalKit",
    "EmergencyRaft"
]


# --------------------------------------------------
# 2. NODE ENCODER
# --------------------------------------------------

def encode_nodes(G):
    """
    Convert NetworkX graph nodes into numerical PyTorch tensors.

    Returns:
        Dictionary containing:
        - Node feature tensors
        - Node IDs in the exact tensor order
    """

    # Separate nodes according to their types.
    incident_nodes = []
    device_nodes = []
    storage_nodes = []
    vehicle_nodes = []

    for node_id, attributes in G.nodes(data=True):

        node_type = attributes["node_type"]

        if node_type == "incident":
            incident_nodes.append((node_id, attributes))

        elif node_type == "device":
            device_nodes.append((node_id, attributes))

        elif node_type == "storage":
            storage_nodes.append((node_id, attributes))

        elif node_type == "vehicle":
            vehicle_nodes.append((node_id, attributes))

    # --------------------------------------------------
    # A. INCIDENT FEATURES
    # --------------------------------------------------
    # Features: x, y, z

    incident_features = []

    for node_id, attributes in incident_nodes:

        features = [
            attributes["x"],
            attributes["y"],
            attributes["z"]
        ]

        incident_features.append(features)

    incident_tensor = torch.tensor(
        incident_features,
        dtype=torch.float
    ).reshape(-1, 3)

    # --------------------------------------------------
    # B. DEVICE FEATURES
    # --------------------------------------------------
    # Features:
    # 4 device-type indicators
    # 1 operating cost
    # 3 dimensions

    device_features = []

    for node_id, attributes in device_nodes:

        device_type = attributes["device_type"]

        # One-hot encoding of the device type.
        type_features = [
            1.0 if device_type == known_type else 0.0
            for known_type in DEVICE_TYPES
        ]

        features = type_features + [
            attributes["operating_cost"],
            attributes["length"],
            attributes["width"],
            attributes["height"]
        ]

        device_features.append(features)

    device_tensor = torch.tensor(
        device_features,
        dtype=torch.float
    ).reshape(-1, 8)

    # --------------------------------------------------
    # C. STORAGE FEATURES
    # --------------------------------------------------
    # Features: x, y, z, number of stored devices

    storage_features = []

    for node_id, attributes in storage_nodes:

        # Count outgoing "stores" relationships.
        device_count = sum(
            1
            for _, _, edge_attributes
            in G.out_edges(node_id, data=True)
            if edge_attributes.get("relation") == "stores"
        )

        features = [
            attributes["x"],
            attributes["y"],
            attributes["z"],
            float(device_count)
        ]

        storage_features.append(features)

    storage_tensor = torch.tensor(
        storage_features,
        dtype=torch.float
    ).reshape(-1, 4)

    # --------------------------------------------------
    # D. VEHICLE FEATURES
    # --------------------------------------------------
    # Features: x, y, z, loading length, width, height

    vehicle_features = []

    for node_id, attributes in vehicle_nodes:

        features = [
            attributes["x"],
            attributes["y"],
            attributes["z"],
            attributes["loading_length"],
            attributes["loading_width"],
            attributes["loading_height"]
        ]

        vehicle_features.append(features)

    vehicle_tensor = torch.tensor(
        vehicle_features,
        dtype=torch.float
    ).reshape(-1, 6)

    # --------------------------------------------------
    # 3. RETURN ENCODED FEATURES AND NODE IDs
    # --------------------------------------------------

    encoded_data = {
        "incident": {
            "x": incident_tensor,
            "node_ids": [node_id for node_id, _ in incident_nodes]
        },

        "device": {
            "x": device_tensor,
            "node_ids": [node_id for node_id, _ in device_nodes]
        },

        "storage": {
            "x": storage_tensor,
            "node_ids": [node_id for node_id, _ in storage_nodes]
        },

        "vehicle": {
            "x": vehicle_tensor,
            "node_ids": [node_id for node_id, _ in vehicle_nodes]
        }
    }

    return encoded_data


# --------------------------------------------------
# 4. DISPLAY ENCODING RESULTS
# --------------------------------------------------

def print_encoding_summary(encoded_data):

    print("\n" + "=" * 55)
    print("DYNAMIC NODE ENCODING RESULTS")
    print("=" * 55)

    for node_type, information in encoded_data.items():

        tensor = information["x"]
        node_ids = information["node_ids"]

        print(f"\n{node_type.upper()}")
        print("-" * 30)

        print("Node IDs:", node_ids)
        print("Feature tensor:")
        print(tensor)

        print("Tensor shape:", tuple(tensor.shape))
        print("Data type:", tensor.dtype)

    print("\n" + "=" * 55)


# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    from src.graph.build_graph import build_graph_from_json

    scenario_file = "data/sample/Scenario_001.json"

    # Build graph from the generated scenario.
    graph = build_graph_from_json(scenario_file)

    # Encode all graph nodes.
    encoded_data = encode_nodes(graph)

    # Display results.
    print_encoding_summary(encoded_data)
