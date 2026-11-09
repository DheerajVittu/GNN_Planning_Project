
import torch
from torch_geometric.data import HeteroData

from src.graph.build_graph import build_graph_from_json
from src.encoder.node_encoder import encode_nodes
from src.encoder.edge_encoder import encode_edges


# --------------------------------------------------
# 1. CREATE HETEROGENEOUS GRAPH
# --------------------------------------------------

def create_hetero_graph(scenario_file):

    # Build the NetworkX graph from JSON.
    networkx_graph = build_graph_from_json(scenario_file)

    # Encode node features.
    encoded_nodes = encode_nodes(networkx_graph)

    # Encode relationships.
    encoded_edges = encode_edges(
        networkx_graph,
        encoded_nodes
    )

    # Create PyTorch Geometric heterogeneous graph.
    data = HeteroData()

    # --------------------------------------------------
    # STEP A: ADD NODE FEATURES
    # --------------------------------------------------

    for node_type, information in encoded_nodes.items():

        data[node_type].x = information["x"]

    # Preserve original IDs in exactly the same order as x.
        data[node_type].node_ids = information["node_ids"]

    # --------------------------------------------------
    # STEP B: ADD EDGE RELATIONSHIPS
    # --------------------------------------------------

    for edge_type, information in encoded_edges.items():

        data[edge_type].edge_index = information["edge_index"]

        if "edge_attr" in information:

            data[edge_type].edge_attr = information["edge_attr"]

    # --------------------------------------------------
    # STEP C: ADD REVERSE STORAGE RELATIONSHIP
    # --------------------------------------------------
    # This allows storage nodes to receive messages
    # from connected device nodes during message passing.

    stores_edge_index = encoded_edges[
        ("storage", "stores", "device")
    ]["edge_index"]

    data[
        "device", "stored_at", "storage"
    ].edge_index = stores_edge_index.flip(0)

    # --------------------------------------------------
    # STEP D: VALIDATE GRAPH
    # --------------------------------------------------

    is_valid = data.validate()

    if not is_valid:
        raise ValueError("Heterogeneous graph validation failed.")

    return data


# --------------------------------------------------
# 2. DISPLAY GRAPH INFORMATION
# --------------------------------------------------

def print_hetero_graph_summary(data):

    print("\n" + "=" * 60)
    print("COMPLETE HETEROGENEOUS GRAPH")
    print("=" * 60)

    print("\nNODE TYPES")
    print("-" * 35)

    print(data.node_types)

    print("\nEDGE TYPES")
    print("-" * 35)

    for edge_type in data.edge_types:
        print(edge_type)

    print("\nNODE FEATURE SHAPES")
    print("-" * 35)

    for node_type in data.node_types:

        print(
            node_type,
            tuple(data[node_type].x.shape)
        )

        print(
            "Node IDs:",
            data[node_type].node_ids
        )

    print("\nEDGE INDEX SHAPES")
    print("-" * 35)

    for edge_type in data.edge_types:

        print(
            edge_type,
            tuple(data[edge_type].edge_index.shape)
        )

    print("\nEDGE ATTRIBUTE SHAPES")
    print("-" * 35)

    for edge_type in data.edge_types:

        if "edge_attr" in data[edge_type]:

            print(
                edge_type,
                tuple(data[edge_type].edge_attr.shape)
            )

    print("\nGRAPH SUMMARY")
    print("-" * 35)

    print(data)

    print("\nGRAPH VALIDATION")
    print("-" * 35)

    print("Is graph valid:", data.validate())

    print("\n" + "=" * 60)


# --------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    scenario_file = "data/sample/Scenario_001.json"

    # Create complete heterogeneous graph.
    hetero_graph = create_hetero_graph(scenario_file)

    # Display graph information.
    print_hetero_graph_summary(hetero_graph)
