
from src.graph.create_hetero_graph import create_hetero_graph
from src.encoder.node_encoder import encode_nodes
from src.encoder.edge_encoder import encode_edges
from src.encoder.normalization import FeatureNormalizer


def create_normalized_hetero_graph(scenario_file, normalizer):
    """
    Create a PyTorch Geometric heterogeneous graph
    and normalize its node and edge features.
    """

    # Step 1: Create the original heterogeneous graph
    graph = create_hetero_graph(scenario_file)

    # Step 2: Rebuild the encoded node and edge features
    # using the same scenario represented by the graph.
    from src.graph.build_graph import build_graph_from_json

    networkx_graph = build_graph_from_json(scenario_file)

    encoded_nodes = encode_nodes(networkx_graph)
    encoded_edges = encode_edges(networkx_graph, encoded_nodes)

    # Step 3: Normalize node and edge features
    normalized_nodes, normalized_edges = normalizer.transform(
        encoded_nodes,
        encoded_edges
    )

    # Step 4: Replace node features in the PyG graph
    for node_type, features in normalized_nodes.items():
        graph[node_type].x = features["x"]

    # Step 5: Replace edge attributes in the PyG graph
    for edge_type, features in normalized_edges.items():
        if "edge_attr" in features:
            graph[edge_type].edge_attr = features["edge_attr"]

    return graph


if __name__ == "__main__":

    scenario_file = "data/sample/Scenario_001.json"

    # Build the raw graph and extract features
    from src.graph.build_graph import build_graph_from_json

    networkx_graph = build_graph_from_json(scenario_file)
    encoded_nodes = encode_nodes(networkx_graph)
    encoded_edges = encode_edges(networkx_graph, encoded_nodes)

    # Temporary demonstration:
    # Fit on this sample only to test the pipeline.
    normalizer = FeatureNormalizer()
    normalizer.fit(
        [encoded_nodes],
        [encoded_edges]
    )

    # Create normalized graph
    graph = create_normalized_hetero_graph(
        scenario_file,
        normalizer
    )

    print("\nNORMALIZED HETEROGENEOUS GRAPH")
    print("=" * 55)

    for node_type in graph.node_types:
        print(
            f"{node_type}: "
            f"{tuple(graph[node_type].x.shape)}"
        )

    for edge_type in graph.edge_types:
        edge_data = graph[edge_type]

        print(
            f"{edge_type}: "
            f"edge_index={tuple(edge_data.edge_index.shape)}"
        )

        if "edge_attr" in edge_data:
            print(
                f"  edge_attr={tuple(edge_data.edge_attr.shape)}"
            )

    print("\nGraph validation:", graph.validate())
    print("\nNormalized graph created successfully!")
