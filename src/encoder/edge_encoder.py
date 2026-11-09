
import torch


# --------------------------------------------------
# 1. DYNAMIC EDGE ENCODER
# --------------------------------------------------

def encode_edges(G, encoded_nodes):
    """
    Convert NetworkX graph relationships into PyTorch tensors.

    Node indices are obtained from the exact ordering
    returned by node_encoder.py.

    Returns:
        Dictionary containing edge indices and attributes.
    """

    # --------------------------------------------------
    # STEP A: CREATE NODE INDEX MAPPINGS
    # --------------------------------------------------

    node_indices = {}

    for node_type, information in encoded_nodes.items():

        node_ids = information["node_ids"]

        node_indices[node_type] = {
            node_id: index
            for index, node_id in enumerate(node_ids)
        }

    # --------------------------------------------------
    # STEP B: INITIALIZE EDGE COLLECTIONS
    # --------------------------------------------------

    stores_edges = []
    requires_edges = []
    requires_attributes = []

    transport_edges = []

    serve_edges = []
    serve_attributes = []

    # --------------------------------------------------
    # STEP C: READ NETWORKX GRAPH RELATIONSHIPS
    # --------------------------------------------------

    for source, target, attributes in G.edges(data=True):

        relation = attributes["relation"]

        # ----------------------------------------------
        # STORAGE -> DEVICE
        # ----------------------------------------------

        if relation == "stores":

            source_index = node_indices["storage"][source]
            target_index = node_indices["device"][target]

            stores_edges.append([
                source_index,
                target_index
            ])

        # ----------------------------------------------
        # INCIDENT -> DEVICE
        # ----------------------------------------------

        elif relation == "requires":

            source_index = node_indices["incident"][source]
            target_index = node_indices["device"][target]

            requires_edges.append([
                source_index,
                target_index
            ])

            requires_attributes.append([
                float(attributes["quantity"])
            ])

        # ----------------------------------------------
        # DEVICE -> VEHICLE
        # ----------------------------------------------

        elif relation == "can_be_transported_by":

            source_index = node_indices["device"][source]
            target_index = node_indices["vehicle"][target]

            transport_edges.append([
                source_index,
                target_index
            ])

        # ----------------------------------------------
        # VEHICLE -> INCIDENT
        # ----------------------------------------------

        elif relation == "can_serve":

            source_index = node_indices["vehicle"][source]
            target_index = node_indices["incident"][target]

            serve_edges.append([
                source_index,
                target_index
            ])

            serve_attributes.append([
                float(attributes["distance"])
            ])

    # --------------------------------------------------
    # STEP D: CONVERT EDGE LISTS INTO TENSORS
    # --------------------------------------------------

    def make_edge_index(edge_list):

        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()

    stores_edge_index = make_edge_index(stores_edges)

    requires_edge_index = make_edge_index(requires_edges)

    transport_edge_index = make_edge_index(transport_edges)

    serve_edge_index = make_edge_index(serve_edges)

    # Edge attributes
    requires_edge_attr = torch.tensor(
        requires_attributes,
        dtype=torch.float
    ).reshape(-1, 1)

    serve_edge_attr = torch.tensor(
        serve_attributes,
        dtype=torch.float
    ).reshape(-1, 1)

    # --------------------------------------------------
    # STEP E: RETURN ALL EDGE INFORMATION
    # --------------------------------------------------

    encoded_edges = {

        ("storage", "stores", "device"): {
            "edge_index": stores_edge_index
        },

        ("incident", "requires", "device"): {
            "edge_index": requires_edge_index,
            "edge_attr": requires_edge_attr
        },

        ("device", "can_be_transported_by", "vehicle"): {
            "edge_index": transport_edge_index
        },

        ("vehicle", "can_serve", "incident"): {
            "edge_index": serve_edge_index,
            "edge_attr": serve_edge_attr
        }
    }

    return encoded_edges


# --------------------------------------------------
# 2. DISPLAY EDGE ENCODING RESULTS
# --------------------------------------------------

def print_edge_summary(encoded_edges):

    print("\n" + "=" * 60)
    print("DYNAMIC EDGE ENCODING RESULTS")
    print("=" * 60)

    for edge_type, information in encoded_edges.items():

        edge_index = information["edge_index"]

        print(f"\n{edge_type}")
        print("-" * 40)

        print("Edge index:")
        print(edge_index)

        print("Edge index shape:", tuple(edge_index.shape))

        if "edge_attr" in information:

            print("Edge attributes:")
            print(information["edge_attr"])

            print(
                "Edge attribute shape:",
                tuple(information["edge_attr"].shape)
            )

    print("\n" + "=" * 60)


# --------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    from src.graph.build_graph import build_graph_from_json
    from src.encoder.node_encoder import encode_nodes

    scenario_file = "data/sample/Scenario_001.json"

    # Build the dynamic NetworkX graph.
    graph = build_graph_from_json(scenario_file)

    # Encode graph nodes first.
    encoded_nodes = encode_nodes(graph)

    # Encode graph relationships using the same node ordering.
    encoded_edges = encode_edges(graph, encoded_nodes)

    # Display edge encoding results.
    print_edge_summary(encoded_edges)
