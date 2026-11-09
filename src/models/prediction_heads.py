
import torch
import torch.nn as nn

from src.graph.create_normalized_hetero_graph import (
    create_normalized_hetero_graph
)
from src.encoder.normalization import FeatureNormalizer
from src.graph.build_graph import build_graph_from_json
from src.encoder.node_encoder import encode_nodes
from src.encoder.edge_encoder import encode_edges
from src.models.hetero_gnn_edge_aware import EdgeAwareHeterogeneousGNN


class AssignmentPredictionHead(nn.Module):
    """
    Calculates a compatibility score for pairs of nodes.

    The score is a logit. Apply sigmoid to convert it
    into a value between 0 and 1.
    """

    def __init__(self, hidden_channels):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, source_embeddings, target_embeddings, edge_index):

        source_indices = edge_index[0]
        target_indices = edge_index[1]

        source = source_embeddings[source_indices]
        target = target_embeddings[target_indices]

        # Combine both node embeddings and their interaction.
        pair_features = torch.cat(
            [source, target, source * target],
            dim=1
        )

        logits = self.network(pair_features).squeeze(-1)

        return logits


class AssignmentPredictionHeads(nn.Module):
    """
    Two prediction heads:

    1. Device -> Vehicle
    2. Vehicle -> Incident
    """

    def __init__(self, hidden_channels=16):
        super().__init__()

        self.device_vehicle_head = AssignmentPredictionHead(
            hidden_channels
        )

        self.vehicle_incident_head = AssignmentPredictionHead(
            hidden_channels
        )

    def forward(self, embeddings, graph):

        # Device -> Vehicle candidate relationships
        device_vehicle_edges = graph[
            "device",
            "can_be_transported_by",
            "vehicle"
        ].edge_index

        device_vehicle_logits = self.device_vehicle_head(
            embeddings["device"],
            embeddings["vehicle"],
            device_vehicle_edges
        )

        # Vehicle -> Incident candidate relationships
        vehicle_incident_edges = graph[
            "vehicle",
            "can_serve",
            "incident"
        ].edge_index

        vehicle_incident_logits = self.vehicle_incident_head(
            embeddings["vehicle"],
            embeddings["incident"],
            vehicle_incident_edges
        )

        return {
            "device_vehicle_logits": device_vehicle_logits,
            "vehicle_incident_logits": vehicle_incident_logits
        }


if __name__ == "__main__":

    scenario_file = "data/sample/Scenario_001.json"

    # Fit the normalizer for this temporary pipeline test.
    networkx_graph = build_graph_from_json(scenario_file)

    encoded_nodes = encode_nodes(networkx_graph)
    encoded_edges = encode_edges(
        networkx_graph,
        encoded_nodes
    )

    normalizer = FeatureNormalizer()
    normalizer.fit(
        [encoded_nodes],
        [encoded_edges]
    )

    # Create normalized graph.
    graph = create_normalized_hetero_graph(
        scenario_file,
        normalizer
    )

    # Initialize the edge-aware GNN.
    input_dimensions = {
        node_type: graph[node_type].x.shape[1]
        for node_type in graph.node_types
    }

    gnn = EdgeAwareHeterogeneousGNN(
        metadata=graph.metadata(),
        input_dimensions=input_dimensions,
        hidden_channels=16
    )

    # Generate node embeddings.
    gnn.eval()

    with torch.no_grad():
        embeddings = gnn(graph)

    # Initialize prediction heads.
    prediction_heads = AssignmentPredictionHeads(
        hidden_channels=16
    )

    prediction_heads.eval()

    # Generate candidate assignment scores.
    with torch.no_grad():
        predictions = prediction_heads(
            embeddings,
            graph
        )

    print("\nASSIGNMENT PREDICTION RESULTS")
    print("=" * 55)

    for prediction_name, logits in predictions.items():

        probabilities = torch.sigmoid(logits)

        print(f"\n{prediction_name}")
        print("Number of candidate assignments:", logits.shape[0])
        print("Logits shape:", tuple(logits.shape))
        print("Probabilities shape:", tuple(probabilities.shape))
        print("Probabilities:")
        print(probabilities)

    print("\nPrediction heads executed successfully!")
