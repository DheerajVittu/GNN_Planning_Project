
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

from src.solution.assignment_decoder import decode_assignments
from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)
from src.solution.objective_function import calculate_objective
from src.solution.constraint_validator import validate_solution


DATA_DIR = Path("data/generated")
CHECKPOINT_PATH = Path("models/checkpoints/edge_aware_gnn_best.pt")

TRAIN_SEEDS = [42, 43, 44, 45, 46, 47, 49, 51, 52, 53]
TEST_SEEDS = [48, 50]


def get_file_paths(seed):

    if seed == 42:
        name = "Scenario_D20_V4_I3"
    else:
        name = f"Scenario_D20_V4_I3_S{seed}"

    scenario_path = DATA_DIR / "scenarios" / f"{name}.json"
    reference_path = DATA_DIR / "references" / f"{name}_reference.json"

    return scenario_path, reference_path


def reconstruct_reference_solution(reference):

    return PlanningSolution(
        solution_id=reference["solution_id"],
        scenario_id=reference["scenario_id"],
        device_assignments=[
            DeviceAssignment(**item)
            for item in reference["device_assignments"]
        ],
        vehicle_assignments=[
            VehicleAssignment(**item)
            for item in reference["vehicle_assignments"]
        ]
    )


def calculate_demand_metrics(scenario, solution):

    vehicle_to_incident = {
        assignment.vehicle_id: assignment.incident_id
        for assignment in solution.vehicle_assignments
    }

    devices = {
        device["device_id"]: device
        for device in scenario["devices"]
    }

    delivered = {
        incident["incident_id"]: {}
        for incident in scenario["incidents"]
    }

    for assignment in solution.device_assignments:

        incident_id = vehicle_to_incident.get(
            assignment.vehicle_id
        )

        if incident_id is None:
            continue

        device_type = devices[assignment.device_id]["device_type"]

        if device_type == "Radio":
            continue

        delivered[incident_id][device_type] = (
            delivered[incident_id].get(device_type, 0) + 1
        )

    total_required = 0
    total_fulfilled = 0
    incidents_served = 0

    for incident in scenario["incidents"]:

        incident_id = incident["incident_id"]

        requirements = {
            device_type: quantity
            for device_type, quantity
            in incident["required_devices"].items()
            if device_type != "Radio"
        }

        incident_fulfilled = 0

        for device_type, required in requirements.items():

            assigned = delivered[incident_id].get(
                device_type, 0
            )

            fulfilled = min(required, assigned)

            total_required += required
            total_fulfilled += fulfilled
            incident_fulfilled += fulfilled

        if incident_fulfilled > 0:
            incidents_served += 1

    coverage_percentage = (
        incidents_served / len(scenario["incidents"]) * 100
    )

    fulfillment_percentage = (
        total_fulfilled / total_required * 100
        if total_required > 0 else 100.0
    )

    return {
        "incidents_served": incidents_served,
        "total_incidents": len(scenario["incidents"]),
        "coverage_percentage": coverage_percentage,
        "total_required": total_required,
        "total_fulfilled": total_fulfilled,
        "unmet": total_required - total_fulfilled,
        "fulfillment_percentage": fulfillment_percentage
    }


def evaluate_solution(scenario, solution):

    objective = calculate_objective(
        scenario,
        solution,
        distance_weight=1.0,
        operating_cost_weight=1.0
    )

    demand = calculate_demand_metrics(
        scenario,
        solution
    )

    validation = validate_solution(
        scenario,
        solution
    )

    return {
        "objective": objective,
        "demand": demand,
        "valid": validation["valid"],
        "hard_constraints_passed": validation[
            "hard_constraints_passed"
        ]
    }


def main():

    print("\nLOADING TRAINED GNN")
    print("=" * 65)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["validation_loss"])

    # Reconstruct the training normalizer.
    training_nodes = []
    training_edges = []

    for seed in TRAIN_SEEDS:

        scenario_path, _ = get_file_paths(seed)

        networkx_graph = build_graph_from_json(
            str(scenario_path)
        )

        encoded_nodes = encode_nodes(networkx_graph)
        encoded_edges = encode_edges(
            networkx_graph,
            encoded_nodes
        )

        training_nodes.append(encoded_nodes)
        training_edges.append(encoded_edges)

    normalizer = FeatureNormalizer()
    normalizer.fit(training_nodes, training_edges)

    print("Normalizer reconstructed successfully.")

    results = []

    for seed in TEST_SEEDS:

        print("\n" + "=" * 65)
        print(f"COMPARISON: SCENARIO {seed}")
        print("=" * 65)

        scenario_path, reference_path = get_file_paths(seed)

        with open(scenario_path, "r", encoding="utf-8") as file:
            scenario = json.load(file)

        with open(reference_path, "r", encoding="utf-8") as file:
            reference_data = json.load(file)

        # 1. Reconstruct the CP-SAT reference solution.
        reference_solution = reconstruct_reference_solution(
            reference_data["reference_solution"]
        )

        # 2. Generate the GNN solution.
        graph = create_normalized_hetero_graph(
            str(scenario_path),
            normalizer
        )

        gnn = EdgeAwareHeterogeneousGNN(
            metadata=graph.metadata(),
            input_dimensions=checkpoint["input_dimensions"],
            hidden_channels=checkpoint["hidden_channels"]
        )

        prediction_heads = AssignmentPredictionHeads(
            hidden_channels=checkpoint["hidden_channels"]
        )

        gnn.load_state_dict(checkpoint["gnn_state_dict"])
        prediction_heads.load_state_dict(
            checkpoint["prediction_heads_state_dict"]
        )

        gnn.eval()
        prediction_heads.eval()

        with torch.no_grad():

            embeddings = gnn(graph)
            predictions = prediction_heads(embeddings, graph)

        gnn_solution = decode_assignments(
            graph=graph,
            predictions=predictions,
            scenario=scenario,
            solution_id=f"GNN_Comparison_{seed}"
        )

        # 3. Evaluate both solutions using the same functions.
        cpsat_result = evaluate_solution(
            scenario,
            reference_solution
        )

        gnn_result = evaluate_solution(
            scenario,
            gnn_solution
        )

        # Detailed assignment comparison
        print("\nCP-SAT VEHICLE ASSIGNMENTS")

        for assignment in reference_solution.vehicle_assignments:
            print(
                assignment.vehicle_id,
                "->",
                assignment.incident_id
            )

        print("\nGNN VEHICLE ASSIGNMENTS")

        for assignment in gnn_solution.vehicle_assignments:
            print(
                assignment.vehicle_id,
                "->",
                assignment.incident_id
            )

        print("\nCP-SAT DEVICE ASSIGNMENTS")

        for assignment in reference_solution.device_assignments:
            print(
                assignment.device_id,
                "->",
                assignment.vehicle_id
            )

        print("\nGNN DEVICE ASSIGNMENTS")

        for assignment in gnn_solution.device_assignments:
            print(
                assignment.device_id,
                "->",
                assignment.vehicle_id
            )

        print("\nVEHICLE DISTANCES - CP-SAT")

        for vehicle, distance in cpsat_result[
            "objective"
        ]["vehicle_distances"].items():
            print(vehicle, ":", round(distance, 2))

        print("\nVEHICLE DISTANCES - GNN")

        for vehicle, distance in gnn_result[
            "objective"
        ]["vehicle_distances"].items():
            print(vehicle, ":", round(distance, 2))

        results.append({
            "scenario": seed,
            "cpsat": cpsat_result,
            "gnn": gnn_result
        })

        # 4. Display comparison.
        print("\nMETRIC COMPARISON")
        print("-" * 65)

        print(
            f"{'Metric':<30}"
            f"{'CP-SAT':>15}"
            f"{'GNN':>15}"
        )

        print("-" * 65)

        rows = [
            (
                "Incidents served",
                f"{cpsat_result['demand']['incidents_served']}/"
                f"{cpsat_result['demand']['total_incidents']}",
                f"{gnn_result['demand']['incidents_served']}/"
                f"{gnn_result['demand']['total_incidents']}"
            ),
            (
                "Coverage (%)",
                f"{cpsat_result['demand']['coverage_percentage']:.2f}",
                f"{gnn_result['demand']['coverage_percentage']:.2f}"
            ),
            (
                "Devices fulfilled",
                f"{cpsat_result['demand']['total_fulfilled']}/"
                f"{cpsat_result['demand']['total_required']}",
                f"{gnn_result['demand']['total_fulfilled']}/"
                f"{gnn_result['demand']['total_required']}"
            ),
            (
                "Fulfilment (%)",
                f"{cpsat_result['demand']['fulfillment_percentage']:.2f}",
                f"{gnn_result['demand']['fulfillment_percentage']:.2f}"
            ),
            (
                "Travel distance",
                f"{cpsat_result['objective']['total_distance']:.2f}",
                f"{gnn_result['objective']['total_distance']:.2f}"
            ),
            (
                "Operating cost",
                f"{cpsat_result['objective']['total_hourly_operating_cost']:.2f}",
                f"{gnn_result['objective']['total_hourly_operating_cost']:.2f}"
            ),
            (
                "Combined objective",
                f"{cpsat_result['objective']['total_objective']:.2f}",
                f"{gnn_result['objective']['total_objective']:.2f}"
            ),
            (
                "Hard constraints",
                str(cpsat_result["hard_constraints_passed"]),
                str(gnn_result["hard_constraints_passed"])
            )
        ]

        for name, cpsat_value, gnn_value in rows:

            print(
                f"{name:<30}"
                f"{cpsat_value:>15}"
                f"{gnn_value:>15}"
            )

        # 5. Calculate objective gap.
        cpsat_objective = cpsat_result[
            "objective"
        ]["total_objective"]

        gnn_objective = gnn_result[
            "objective"
        ]["total_objective"]

        if cpsat_objective > 0:

            gap = (
                (gnn_objective - cpsat_objective)
                / cpsat_objective
            ) * 100

            print("\nGNN objective gap:", f"{gap:.2f}%")

        print(
            "CP-SAT solution valid:",
            cpsat_result["valid"]
        )

        print(
            "GNN solution valid:",
            gnn_result["valid"]
        )

    # 6. Summary across both scenarios.
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    for result in results:

        seed = result["scenario"]
        cp = result["cpsat"]
        gn = result["gnn"]

        print(f"\nScenario {seed}")

        print(
            "CP-SAT objective:",
            round(cp["objective"]["total_objective"], 2)
        )

        print(
            "GNN objective:",
            round(gn["objective"]["total_objective"], 2)
        )

        print(
            "CP-SAT fulfilled:",
            f"{cp['demand']['total_fulfilled']}/"
            f"{cp['demand']['total_required']}"
        )

        print(
            "GNN fulfilled:",
            f"{gn['demand']['total_fulfilled']}/"
            f"{gn['demand']['total_required']}"
        )

    print("\nCOMPARISON COMPLETED")


if __name__ == "__main__":
    main()
