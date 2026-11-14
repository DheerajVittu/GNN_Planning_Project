
import json
import torch
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

from src.solution.assignment_decoder import decode_assignments
from src.solution.constraint_validator import validate_solution


DATA_DIR = Path("data/generated")
CHECKPOINT_PATH = Path("models/checkpoints/edge_aware_gnn_best.pt")

TRAIN_SEEDS = [42, 43, 44, 45, 46, 47, 49, 51, 52, 53]
TEST_SEEDS = [48, 50]


def get_file_paths(seed):

    if seed == 42:
        scenario_name = "Scenario_D20_V4_I3"
    else:
        scenario_name = f"Scenario_D20_V4_I3_S{seed}"

    scenario_path = DATA_DIR / "scenarios" / f"{scenario_name}.json"
    reference_path = DATA_DIR / "references" / f"{scenario_name}_reference.json"

    return scenario_path, reference_path


def calculate_metrics(logits, labels):

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

    true_positives = ((predictions == 1) & (labels == 1)).sum().item()
    true_negatives = ((predictions == 0) & (labels == 0)).sum().item()
    false_positives = ((predictions == 1) & (labels == 0)).sum().item()
    false_negatives = ((predictions == 0) & (labels == 1)).sum().item()

    total = len(labels)

    accuracy = (true_positives + true_negatives) / total
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)

    f1 = (
        2 * precision * recall / max(precision + recall, 1e-8)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def main():

    print("\nLOADING TRAINED GNN CHECKPOINT")
    print("=" * 55)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("Best training epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["validation_loss"])

    # 1. Reconstruct the training normalizer.
    print("\nRECONSTRUCTING TRAINING NORMALIZER")

    training_nodes = []
    training_edges = []

    for seed in TRAIN_SEEDS:

        scenario_path, _ = get_file_paths(seed)

        networkx_graph = build_graph_from_json(str(scenario_path))

        encoded_nodes = encode_nodes(networkx_graph)
        encoded_edges = encode_edges(
            networkx_graph,
            encoded_nodes
        )

        training_nodes.append(encoded_nodes)
        training_edges.append(encoded_edges)

    normalizer = FeatureNormalizer()

    normalizer.fit(training_nodes, training_edges)

    print("Normalizer fitted on", len(TRAIN_SEEDS), "training scenarios.")

    # 2. Build the first validation graph to initialize the models.
    first_scenario_path, _ = get_file_paths(TEST_SEEDS[0])

    first_graph = create_normalized_hetero_graph(
        str(first_scenario_path),
        normalizer
    )

    input_dimensions = checkpoint["input_dimensions"]
    hidden_channels = checkpoint["hidden_channels"]

    gnn = EdgeAwareHeterogeneousGNN(
        metadata=first_graph.metadata(),
        input_dimensions=input_dimensions,
        hidden_channels=hidden_channels
    )

    prediction_heads = AssignmentPredictionHeads(
        hidden_channels=hidden_channels
    )

    gnn.load_state_dict(checkpoint["gnn_state_dict"])
    prediction_heads.load_state_dict(
        checkpoint["prediction_heads_state_dict"]
    )

    gnn.eval()
    prediction_heads.eval()

    print("Trained GNN and prediction heads loaded successfully.")

    # 3. Evaluate both unseen validation scenarios.
    for seed in TEST_SEEDS:

        print("\n" + "=" * 55)
        print("EVALUATING VALIDATION SCENARIO:", seed)
        print("=" * 55)

        scenario_path, reference_path = get_file_paths(seed)

        with open(scenario_path, "r", encoding="utf-8") as file:
            scenario = json.load(file)

        with open(reference_path, "r", encoding="utf-8") as file:
            reference_data = json.load(file)

        graph = create_normalized_hetero_graph(
            str(scenario_path),
            normalizer
        )

        labels = create_assignment_labels(
            graph,
            reference_data["reference_solution"]
        )

        # 4. Generate trained GNN predictions.
        with torch.no_grad():

            embeddings = gnn(graph)
            predictions = prediction_heads(embeddings, graph)

        # 5. Compare predictions with CP-SAT labels.
        device_metrics = calculate_metrics(
            predictions["device_vehicle_logits"],
            labels["device_vehicle_labels"]
        )

        vehicle_metrics = calculate_metrics(
            predictions["vehicle_incident_logits"],
            labels["vehicle_incident_labels"]
        )

        print("\nDEVICE-TO-VEHICLE PREDICTIONS")
        print("Accuracy:", round(device_metrics["accuracy"], 4))
        print("Precision:", round(device_metrics["precision"], 4))
        print("Recall:", round(device_metrics["recall"], 4))
        print("F1:", round(device_metrics["f1"], 4))

        print("\nVEHICLE-TO-INCIDENT PREDICTIONS")
        print("Accuracy:", round(vehicle_metrics["accuracy"], 4))
        print("Precision:", round(vehicle_metrics["precision"], 4))
        print("Recall:", round(vehicle_metrics["recall"], 4))
        print("F1:", round(vehicle_metrics["f1"], 4))

        # 6. Decode the predictions into a planning solution.
        solution = decode_assignments(
            graph=graph,
            predictions=predictions,
            scenario=scenario,
            solution_id=f"GNN_Validation_{seed}"
        )

        # 7. Validate all hard constraints.
        validation = validate_solution(
            scenario,
            solution
        )

        print("\nDECODED PLANNING SOLUTION")
        print("Assigned devices:", len(solution.device_assignments))
        print("Deployed vehicles:", len(solution.vehicle_assignments))

        print("\nVEHICLE ASSIGNMENTS")

        for assignment in solution.vehicle_assignments:
            print(
                assignment.vehicle_id,
                "->",
                assignment.incident_id
            )

        print("\nHARD CONSTRAINT VALIDATION")

        for name, check in validation["checks"].items():

            status = "PASS" if check["passed"] else "FAIL"

            print(f"{name}: {status}")

        print(
            "\nAll hard constraints passed:",
            validation["hard_constraints_passed"]
        )

        print(
            "Overall solution valid:",
            validation["valid"]
        )

    print("\nGNN EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
