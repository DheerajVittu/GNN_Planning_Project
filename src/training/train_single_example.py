
import json
import torch
import torch.nn as nn

from src.graph.build_graph import build_graph_from_json
from src.encoder.node_encoder import encode_nodes
from src.encoder.edge_encoder import encode_edges
from src.encoder.normalization import FeatureNormalizer
from src.graph.create_normalized_hetero_graph import (
    create_normalized_hetero_graph
)
from src.models.hetero_gnn_edge_aware import EdgeAwareHeterogeneousGNN
from src.models.prediction_heads import AssignmentPredictionHeads
from src.training.label_generator import create_assignment_labels


def main():

    scenario_file = "data/generated/scenarios/Scenario_D20_V4_I3.json"
    reference_file = "data/generated/references/Scenario_D20_V4_I3_reference.json"

    # 1. Load the scenario and encode its features.
    networkx_graph = build_graph_from_json(scenario_file)

    encoded_nodes = encode_nodes(networkx_graph)
    encoded_edges = encode_edges(networkx_graph, encoded_nodes)

    # 2. Fit a temporary normalizer.
    normalizer = FeatureNormalizer()
    normalizer.fit([encoded_nodes], [encoded_edges])

    # 3. Create the normalized heterogeneous graph.
    graph = create_normalized_hetero_graph(
        scenario_file,
        normalizer
    )

    # 4. Load CP-SAT reference assignments.
    with open(reference_file, "r") as file:
        reference_data = json.load(file)

    reference_solution = reference_data["reference_solution"]

    # 5. Generate training labels.
    labels = create_assignment_labels(
        graph,
        reference_solution
    )

    # 6. Initialize the GNN.
    input_dimensions = {
        node_type: graph[node_type].x.shape[1]
        for node_type in graph.node_types
    }

    gnn = EdgeAwareHeterogeneousGNN(
        metadata=graph.metadata(),
        input_dimensions=input_dimensions,
        hidden_channels=16
    )

    prediction_heads = AssignmentPredictionHeads(
        hidden_channels=16
    )

    # 7. Combine both models for optimization.
    parameters = (
        list(gnn.parameters())
        + list(prediction_heads.parameters())
    )

    optimizer = torch.optim.Adam(
        parameters,
        lr=0.001
    )

    # Account for the different numbers of positive and negative labels.
    device_labels = labels["device_vehicle_labels"]
    vehicle_labels = labels["vehicle_incident_labels"]

    device_pos_weight = (
        (device_labels == 0).sum().float()
        / (device_labels == 1).sum().float()
    )

    vehicle_pos_weight = (
        (vehicle_labels == 0).sum().float()
        / (vehicle_labels == 1).sum().float()
    )

    device_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=device_pos_weight
    )

    vehicle_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=vehicle_pos_weight
    )

    print("\nSTARTING SINGLE-EXAMPLE GNN TRAINING")
    print("=" * 55)

    # 8. Training loop.
    for epoch in range(1, 101):

        gnn.train()
        prediction_heads.train()

        optimizer.zero_grad()

        embeddings = gnn(graph)
        predictions = prediction_heads(embeddings, graph)

        device_loss = device_loss_function(
            predictions["device_vehicle_logits"],
            device_labels
        )

        vehicle_loss = vehicle_loss_function(
            predictions["vehicle_incident_logits"],
            vehicle_labels
        )

        total_loss = device_loss + vehicle_loss

        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d}/100 | "
                f"Total Loss: {total_loss.item():.4f} | "
                f"Device Loss: {device_loss.item():.4f} | "
                f"Vehicle Loss: {vehicle_loss.item():.4f}"
            )

    print("\nSINGLE-EXAMPLE TRAINING COMPLETED")
    print("Final loss:", round(total_loss.item(), 4))


if __name__ == "__main__":
    main()
