
import json
import torch
import torch.nn as nn

from pathlib import Path

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


DATA_DIR = Path("data/generated")

TRAIN_SEEDS = [42, 43, 44, 45, 46, 47, 49, 51, 52, 53]
VALIDATION_SEEDS = [48, 50]


def get_file_paths(seed):
    if seed == 42:
        scenario_name = "Scenario_D20_V4_I3"
    else:
        scenario_name = f"Scenario_D20_V4_I3_S{seed}"

    scenario_path = DATA_DIR / "scenarios" / f"{scenario_name}.json"
    reference_path = DATA_DIR / "references" / f"{scenario_name}_reference.json"

    return scenario_path, reference_path


def load_raw_features(seed):
    scenario_path, reference_path = get_file_paths(seed)

    if not scenario_path.exists() or not reference_path.exists():
        raise FileNotFoundError(f"Missing files for seed {seed}")

    networkx_graph = build_graph_from_json(str(scenario_path))

    encoded_nodes = encode_nodes(networkx_graph)
    encoded_edges = encode_edges(networkx_graph, encoded_nodes)

    with open(reference_path, "r", encoding="utf-8") as file:
        reference_data = json.load(file)

    return {
        "seed": seed,
        "scenario_path": str(scenario_path),
        "reference_solution": reference_data["reference_solution"],
        "encoded_nodes": encoded_nodes,
        "encoded_edges": encoded_edges
    }


def prepare_graphs(examples, normalizer):
    prepared = []

    for example in examples:
        graph = create_normalized_hetero_graph(
            example["scenario_path"],
            normalizer
        )

        labels = create_assignment_labels(
            graph,
            example["reference_solution"]
        )

        prepared.append({
            "seed": example["seed"],
            "graph": graph,
            "labels": labels
        })

    return prepared


def main():

    print("\nLOADING MULTI-SCENARIO DATASET")
    print("=" * 55)

    # 1. Load raw features for all scenarios.
    all_seeds = TRAIN_SEEDS + VALIDATION_SEEDS
    examples = [load_raw_features(seed) for seed in all_seeds]

    training_examples = [
        item for item in examples
        if item["seed"] in TRAIN_SEEDS
    ]

    validation_examples = [
        item for item in examples
        if item["seed"] in VALIDATION_SEEDS
    ]

    print("Training scenarios:", len(training_examples))
    print("Validation scenarios:", len(validation_examples))

    # 2. Fit normalization using training scenarios only.
    normalizer = FeatureNormalizer()

    normalizer.fit(
        [item["encoded_nodes"] for item in training_examples],
        [item["encoded_edges"] for item in training_examples]
    )

    # 3. Convert all scenarios into normalized graphs.
    training_data = prepare_graphs(training_examples, normalizer)
    validation_data = prepare_graphs(validation_examples, normalizer)

    print("\nGRAPH DATASET SUMMARY")

    for item in training_data + validation_data:
        graph = item["graph"]
        labels = item["labels"]

        print(
            f"Seed {item['seed']} | "
            f"Device edges: {len(labels['device_vehicle_labels'])} | "
            f"Positive: {int(labels['device_vehicle_labels'].sum())} | "
            f"Vehicle edges: {len(labels['vehicle_incident_labels'])} | "
            f"Positive: {int(labels['vehicle_incident_labels'].sum())}"
        )

    # 4. Calculate class weights from training data only.
    all_device_labels = torch.cat([
        item["labels"]["device_vehicle_labels"]
        for item in training_data
    ])

    all_vehicle_labels = torch.cat([
        item["labels"]["vehicle_incident_labels"]
        for item in training_data
    ])

    device_pos_weight = (
        (all_device_labels == 0).sum().float()
        / (all_device_labels == 1).sum().float()
    )

    vehicle_pos_weight = (
        (all_vehicle_labels == 0).sum().float()
        / (all_vehicle_labels == 1).sum().float()
    )

    device_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=device_pos_weight
    )

    vehicle_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=vehicle_pos_weight
    )

    # 5. Initialize the GNN and prediction heads.
    first_graph = training_data[0]["graph"]

    input_dimensions = {
        node_type: first_graph[node_type].x.shape[1]
        for node_type in first_graph.node_types
    }

    hidden_channels = 16

    gnn = EdgeAwareHeterogeneousGNN(
        metadata=first_graph.metadata(),
        input_dimensions=input_dimensions,
        hidden_channels=hidden_channels
    )

    prediction_heads = AssignmentPredictionHeads(
        hidden_channels=hidden_channels
    )

    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(prediction_heads.parameters()),
        lr=0.001
    )

    # 6. Train across multiple scenarios.
    
    # 6. Train across multiple scenarios.
    epochs = 100
    patience = 15

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_directory = Path("models/checkpoints")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_directory / "edge_aware_gnn_best.pt"
    )

    best_gnn_state = None
    best_heads_state = None

    print("\nSTARTING MULTI-SCENARIO GNN TRAINING")
    print("=" * 55)

    for epoch in range(1, epochs + 1):

        gnn.train()
        prediction_heads.train()

        total_training_loss = 0.0

        # Shuffle scenario order each epoch.
        order = torch.randperm(len(training_data)).tolist()

        for index in order:

            item = training_data[index]
            graph = item["graph"]
            labels = item["labels"]

            optimizer.zero_grad()

            embeddings = gnn(graph)
            predictions = prediction_heads(embeddings, graph)

            device_loss = device_loss_function(
                predictions["device_vehicle_logits"],
                labels["device_vehicle_labels"]
            )

            vehicle_loss = vehicle_loss_function(
                predictions["vehicle_incident_logits"],
                labels["vehicle_incident_labels"]
            )

            loss = device_loss + vehicle_loss

            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

        average_training_loss = (
            total_training_loss / len(training_data)
        )

        # 7. Evaluate on unseen validation scenarios.
        gnn.eval()
        prediction_heads.eval()

        total_validation_loss = 0.0

        with torch.no_grad():

            for item in validation_data:

                graph = item["graph"]
                labels = item["labels"]

                embeddings = gnn(graph)
                predictions = prediction_heads(embeddings, graph)

                device_loss = device_loss_function(
                    predictions["device_vehicle_logits"],
                    labels["device_vehicle_labels"]
                )

                vehicle_loss = vehicle_loss_function(
                    predictions["vehicle_incident_logits"],
                    labels["vehicle_incident_labels"]
                )

                total_validation_loss += (
                    device_loss + vehicle_loss
                ).item()

        average_validation_loss = (
            total_validation_loss / len(validation_data)
        )

        # 8. Save the model whenever validation improves.
        if average_validation_loss < best_validation_loss:

            best_validation_loss = average_validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_gnn_state = {
                key: value.detach().clone()
                for key, value in gnn.state_dict().items()
            }

            best_heads_state = {
                key: value.detach().clone()
                for key, value in prediction_heads.state_dict().items()
            }

            torch.save(
                {
                    "epoch": best_epoch,
                    "validation_loss": best_validation_loss,
                    "gnn_state_dict": best_gnn_state,
                    "prediction_heads_state_dict": best_heads_state,
                    "input_dimensions": input_dimensions,
                    "hidden_channels": hidden_channels
                },
                checkpoint_path
            )

        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Training Loss: {average_training_loss:.4f} | "
                f"Validation Loss: {average_validation_loss:.4f} | "
                f"Best Epoch: {best_epoch}"
            )

        # 9. Stop if validation has not improved for 15 epochs.
        if epochs_without_improvement >= patience:

            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"No validation improvement for {patience} epochs."
            )
            break

    # 10. Restore the best model weights.
    gnn.load_state_dict(best_gnn_state)
    prediction_heads.load_state_dict(best_heads_state)

    print("\nMULTI-SCENARIO TRAINING COMPLETED")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_validation_loss, 4))
    print("Best model saved:", checkpoint_path)


if __name__ == "__main__":
    main()
