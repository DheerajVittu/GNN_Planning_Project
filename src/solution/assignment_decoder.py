import torch

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)


def decode_assignments(
    graph,
    predictions,
    scenario_id,
    solution_id="Solution_001"
):
    """
    Convert GNN prediction scores into a preliminary planning solution.

    Selection rules:
    1. Each vehicle is assigned to its highest-scoring incident.
    2. A device is eligible for a vehicle only if the vehicle's
       selected incident requires that device.
    3. Each device is assigned to at most one vehicle.

    This is a preliminary greedy decoder. It does not yet guarantee
    that all six hard constraints are satisfied.
    """

    solution = PlanningSolution(
        solution_id=solution_id,
        scenario_id=scenario_id
    )

    # Original entity IDs, in the same order as graph node indices.
    device_ids = graph["device"].node_ids
    vehicle_ids = graph["vehicle"].node_ids
    incident_ids = graph["incident"].node_ids

    # Convert logits into probabilities.
    device_vehicle_scores = torch.sigmoid(
        predictions["device_vehicle_logits"]
    ).detach().cpu()

    vehicle_incident_scores = torch.sigmoid(
        predictions["vehicle_incident_logits"]
    ).detach().cpu()

    # Candidate relationships.
    device_vehicle_edges = graph[
        "device",
        "can_be_transported_by",
        "vehicle"
    ].edge_index.detach().cpu()

    vehicle_incident_edges = graph[
        "vehicle",
        "can_serve",
        "incident"
    ].edge_index.detach().cpu()

    required_edges = graph[
        "incident",
        "requires",
        "device"
    ].edge_index.detach().cpu()

    # --------------------------------------------------
    # STEP 1: Assign each vehicle to one incident.
    # --------------------------------------------------

    selected_incidents = {}

    for edge_position in range(vehicle_incident_edges.shape[1]):

        vehicle_index = int(
            vehicle_incident_edges[0, edge_position]
        )

        incident_index = int(
            vehicle_incident_edges[1, edge_position]
        )

        score = float(vehicle_incident_scores[edge_position])

        if (
            vehicle_index not in selected_incidents
            or score > selected_incidents[vehicle_index][1]
        ):
            selected_incidents[vehicle_index] = (
                incident_index,
                score
            )

    for vehicle_index, (incident_index, score) in (
        selected_incidents.items()
    ):
        solution.vehicle_assignments.append(
            VehicleAssignment(
                vehicle_id=vehicle_ids[vehicle_index],
                incident_id=incident_ids[incident_index]
            )
        )

    # --------------------------------------------------
    # STEP 2: Identify which devices are required by
    # the incident selected for each vehicle.
    # --------------------------------------------------

    required_pairs = set()

    for edge_position in range(required_edges.shape[1]):

        incident_index = int(required_edges[0, edge_position])
        device_index = int(required_edges[1, edge_position])

        required_pairs.add((incident_index, device_index))

    # --------------------------------------------------
    # STEP 3: Assign each device to its highest-scoring
    # eligible vehicle.
    # --------------------------------------------------

    for device_index, device_id in enumerate(device_ids):

        best_vehicle = None
        best_score = float("-inf")

        for edge_position in range(device_vehicle_edges.shape[1]):

            candidate_device_index = int(
                device_vehicle_edges[0, edge_position]
            )

            vehicle_index = int(
                device_vehicle_edges[1, edge_position]
            )

            if candidate_device_index != device_index:
                continue

            # Vehicle must already have a selected incident.
            if vehicle_index not in selected_incidents:
                continue

            incident_index = selected_incidents[vehicle_index][0]

            # The selected incident must require this device.
            if (incident_index, device_index) not in required_pairs:
                continue

            score = float(device_vehicle_scores[edge_position])

            if score > best_score:
                best_score = score
                best_vehicle = vehicle_index

        if best_vehicle is not None:
            solution.device_assignments.append(
                DeviceAssignment(
                    device_id=device_id,
                    vehicle_id=vehicle_ids[best_vehicle]
                )
            )

    return solution