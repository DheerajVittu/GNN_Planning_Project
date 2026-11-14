
import torch
from collections import defaultdict

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)


def _sorted_device_dimensions(device):
    return sorted([
        device["length"],
        device["width"],
        device["height"]
    ])


def _sorted_vehicle_dimensions(vehicle):
    return sorted([
        vehicle["loading_length"],
        vehicle["loading_width"],
        vehicle["loading_height"]
    ])


def _can_add_device(device, vehicle, assigned_devices, device_by_id):
    """
    Check individual fit and simplified combined loading capacity.

    Uses the same provisional rotation assumption as the validator.
    """

    device_dimensions = _sorted_device_dimensions(device)
    vehicle_dimensions = _sorted_vehicle_dimensions(vehicle)

    # Check individual fit.
    if any(
        device_dimensions[i] > vehicle_dimensions[i]
        for i in range(3)
    ):
        return False

    # Calculate dimensions of devices already assigned.
    total_dimensions = [0, 0, 0]

    for device_id in assigned_devices:
        existing_device = device_by_id[device_id]
        dimensions = _sorted_device_dimensions(existing_device)

        for i in range(3):
            total_dimensions[i] += dimensions[i]

    # Add the new device.
    for i in range(3):
        total_dimensions[i] += device_dimensions[i]

    return all(
        total_dimensions[i] <= vehicle_dimensions[i]
        for i in range(3)
    )


def decode_assignments(
    graph,
    predictions,
    scenario,
    solution_id="Solution_001"
):
    """
    Constraint-aware greedy decoder.

    Inputs:
        graph: PyTorch Geometric heterogeneous graph.
        predictions: GNN prediction dictionary.
        scenario: Original scenario dictionary loaded from JSON.

    Returns:
        PlanningSolution

    This is a preliminary feasible-assignment decoder,
    not an optimizer for the complete planning objective.
    """

    solution = PlanningSolution(
        solution_id=solution_id,
        scenario_id=scenario["scenario_id"]
    )

    # --------------------------------------------------
    # STEP 1: ORIGINAL SCENARIO INFORMATION
    # --------------------------------------------------

    devices = scenario["devices"]
    vehicles = scenario["vehicles"]
    incidents = scenario["incidents"]
    storages = scenario["storages"]

    device_by_id = {
        device["device_id"]: device
        for device in devices
    }

    vehicle_by_id = {
        vehicle["vehicle_id"]: vehicle
        for vehicle in vehicles
    }

    incident_by_id = {
        incident["incident_id"]: incident
        for incident in incidents
    }

    # Available devices from storage inventories.
    available_device_ids = set()

    for storage in storages:
        available_device_ids.update(storage.get("device_ids", []))

    # --------------------------------------------------
    # STEP 2: GRAPH NODE IDS AND PREDICTION SCORES
    # --------------------------------------------------

    device_ids = graph["device"].node_ids
    vehicle_ids = graph["vehicle"].node_ids
    incident_ids = graph["incident"].node_ids

    device_vehicle_scores = torch.sigmoid(
        predictions["device_vehicle_logits"]
    ).detach().cpu().reshape(-1)

    vehicle_incident_scores = torch.sigmoid(
        predictions["vehicle_incident_logits"]
    ).detach().cpu().reshape(-1)

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

    # --------------------------------------------------
    # STEP 3: CREATE SCORE LOOKUPS
    # --------------------------------------------------

    vehicle_incident_candidates = []

    for position in range(vehicle_incident_edges.shape[1]):

        vehicle_index = int(vehicle_incident_edges[0, position])
        incident_index = int(vehicle_incident_edges[1, position])

        score = float(vehicle_incident_scores[position])

        vehicle_incident_candidates.append((
            score,
            vehicle_index,
            incident_index
        ))

    # Highest-scoring vehicle-incident pairs first.
    vehicle_incident_candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    device_vehicle_score_lookup = {}

    for position in range(device_vehicle_edges.shape[1]):

        device_index = int(device_vehicle_edges[0, position])
        vehicle_index = int(device_vehicle_edges[1, position])

        score = float(device_vehicle_scores[position])

        device_vehicle_score_lookup[
            (device_index, vehicle_index)
        ] = score

    # --------------------------------------------------
    # STEP 4: SELECT VEHICLE-INCIDENT ASSIGNMENTS
    # --------------------------------------------------

    
    # --------------------------------------------------
    # STEP 4: COVERAGE-FIRST VEHICLE-INCIDENT SELECTION
    # --------------------------------------------------

    selected_incidents = {}
    vehicles_per_incident = defaultdict(list)

    available_vehicle_indices = set(range(len(vehicle_ids)))

    # First pass:
    # Give each incident an opportunity to receive one vehicle.
    # For each incident, select its highest-scoring available vehicle.

    for incident_index, incident_id in enumerate(incident_ids):

        candidates = [
            (score, vehicle_index)
            for score, vehicle_index, candidate_incident_index
            in vehicle_incident_candidates
            if candidate_incident_index == incident_index
            and vehicle_index in available_vehicle_indices
        ]

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        if not candidates:
            continue

        score, selected_vehicle_index = candidates[0]

        selected_incidents[selected_vehicle_index] = incident_index

        vehicles_per_incident[incident_id].append(
            selected_vehicle_index
        )

        available_vehicle_indices.remove(
            selected_vehicle_index
        )

        solution.vehicle_assignments.append(
            VehicleAssignment(
                vehicle_id=vehicle_ids[selected_vehicle_index],
                incident_id=incident_id
            )
        )


    # --------------------------------------------------
    # STEP 5: ASSIGN ONE RADIO TO EACH DEPLOYED VEHICLE
    # --------------------------------------------------

    available_radios = [
        device for device in devices
        if device["device_type"].lower() == "radio"
        and device["device_id"] in available_device_ids
    ]

    assigned_device_ids = set()
    assigned_devices_by_vehicle = defaultdict(list)

    deployed_vehicle_indices = list(selected_incidents.keys())

    # Prioritize vehicles with higher vehicle-incident scores.
    selected_incidents_ordered = sorted(
        deployed_vehicle_indices,
        key=lambda index: next(
            score for score, v_index, _ in vehicle_incident_candidates
            if v_index == index
        ),
        reverse=True
    )

    for vehicle_index in selected_incidents_ordered:

        vehicle_id = vehicle_ids[vehicle_index]
        vehicle = vehicle_by_id[vehicle_id]

        radio_candidates = []

        for radio in available_radios:

            radio_id = radio["device_id"]

            if radio_id in assigned_device_ids:
                continue

            if not _can_add_device(
                radio,
                vehicle,
                assigned_devices_by_vehicle[vehicle_id],
                device_by_id
            ):
                continue

            try:
                radio_index = device_ids.index(radio_id)
            except ValueError:
                continue

            score = device_vehicle_score_lookup.get(
                (radio_index, vehicle_index),
                float("-inf")
            )

            if score != float("-inf"):
                radio_candidates.append((score, radio))

        if not radio_candidates:
            # Do not deploy a vehicle without a Radio.
            # Remove its incident assignment as well.
            selected_incidents.pop(vehicle_index)

            incident_id = next(
                assignment.incident_id
                for assignment in solution.vehicle_assignments
                if assignment.vehicle_id == vehicle_id
            )

            solution.vehicle_assignments = [
                assignment
                for assignment in solution.vehicle_assignments
                if assignment.vehicle_id != vehicle_id
            ]

            vehicles_per_incident[incident_id].remove(vehicle_index)
            continue

        radio_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        selected_radio = radio_candidates[0][1]
        radio_id = selected_radio["device_id"]

        solution.device_assignments.append(
            DeviceAssignment(
                device_id=radio_id,
                vehicle_id=vehicle_id
            )
        )

        assigned_device_ids.add(radio_id)
        assigned_devices_by_vehicle[vehicle_id].append(radio_id)

    # --------------------------------------------------
    # STEP 6: ALLOCATE INCIDENT REQUIREMENTS
    # --------------------------------------------------

    for incident in incidents:

        incident_id = incident["incident_id"]

        assigned_vehicle_indices = vehicles_per_incident.get(
            incident_id, []
        )

        # Keep only vehicles that successfully received a Radio.
        assigned_vehicle_indices = [
            index for index in assigned_vehicle_indices
            if index in selected_incidents
        ]

        if not assigned_vehicle_indices:
            continue

        required_devices = incident["required_devices"]

        for device_type, required_quantity in required_devices.items():

            # Radio is handled separately as mandatory vehicle equipment.
            if device_type.lower() == "radio":
                continue

            candidates = []

            for device in devices:

                device_id = device["device_id"]

                if device["device_type"] != device_type:
                    continue

                if device_id not in available_device_ids:
                    continue

                if device_id in assigned_device_ids:
                    continue

                try:
                    device_index = device_ids.index(device_id)
                except ValueError:
                    continue

                for vehicle_index in assigned_vehicle_indices:

                    vehicle_id = vehicle_ids[vehicle_index]
                    vehicle = vehicle_by_id[vehicle_id]

                    if not _can_add_device(
                        device,
                        vehicle,
                        assigned_devices_by_vehicle[vehicle_id],
                        device_by_id
                    ):
                        continue

                    score = device_vehicle_score_lookup.get(
                        (device_index, vehicle_index),
                        float("-inf")
                    )

                    if score != float("-inf"):
                        candidates.append((
                            score,
                            device_id,
                            vehicle_id
                        ))

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            allocated = 0

            for score, device_id, vehicle_id in candidates:

                if allocated >= required_quantity:
                    break

                if device_id in assigned_device_ids:
                    continue

                device = device_by_id[device_id]
                vehicle = vehicle_by_id[vehicle_id]

                # Recheck capacity because previous assignments
                # may have consumed some loading space.
                if not _can_add_device(
                    device,
                    vehicle,
                    assigned_devices_by_vehicle[vehicle_id],
                    device_by_id
                ):
                    continue

                solution.device_assignments.append(
                    DeviceAssignment(
                        device_id=device_id,
                        vehicle_id=vehicle_id
                    )
                )

                assigned_device_ids.add(device_id)
                assigned_devices_by_vehicle[vehicle_id].append(
                    device_id
                )

                allocated += 1


    # --------------------------------------------------
    # STEP 7: DEPLOY ADDITIONAL VEHICLES WHEN NEEDED
    # --------------------------------------------------

    # The first allocation pass may leave some incident
    # requirements unfulfilled. Try assigning additional
    # vehicles to those incidents.

    while True:

        best_candidate = None
        best_score = float("-inf")

        # Examine all vehicle-incident combinations.
        for vehicle_score, vehicle_index, incident_index in vehicle_incident_candidates:

            # A vehicle can serve only one incident.
            if vehicle_index in selected_incidents:
                continue

            vehicle_id = vehicle_ids[vehicle_index]
            vehicle = vehicle_by_id[vehicle_id]

            incident_id = incident_ids[incident_index]
            incident = incident_by_id[incident_id]

            # Calculate remaining demand for this incident.
            assigned_counts = defaultdict(int)

            for assignment in solution.device_assignments:

                assigned_vehicle_id = assignment.vehicle_id

                assigned_incident_index = selected_incidents.get(
                    vehicle_ids.index(assigned_vehicle_id)
                )

                if assigned_incident_index == incident_index:
                    assigned_device = device_by_id[assignment.device_id]

                    assigned_counts[assigned_device["device_type"]] += 1

            unmet_types = []

            for device_type, required_quantity in incident["required_devices"].items():

                if device_type.lower() == "radio":
                    continue

                remaining = (
                    required_quantity - assigned_counts[device_type]
                )

                if remaining > 0:
                    unmet_types.append(device_type)

            if not unmet_types:
                continue

            # Find an available Radio for the additional vehicle.
            for radio in available_radios:

                radio_id = radio["device_id"]

                if radio_id in assigned_device_ids:
                    continue

                if not _can_add_device(
                    radio,
                    vehicle,
                    [],
                    device_by_id
                ):
                    continue

                radio_index = device_ids.index(radio_id)

                radio_score = device_vehicle_score_lookup.get(
                    (radio_index, vehicle_index),
                    float("-inf")
                )

                if radio_score == float("-inf"):
                    continue

                # Find a required device that can travel
                # with the Radio in this vehicle.
                for device_type in unmet_types:

                    for device in devices:

                        device_id = device["device_id"]

                        if device["device_type"] != device_type:
                            continue

                        if device_id not in available_device_ids:
                            continue

                        if device_id in assigned_device_ids:
                            continue

                        if not _can_add_device(
                            device,
                            vehicle,
                            [radio_id],
                            device_by_id
                        ):
                            continue

                        device_index = device_ids.index(device_id)

                        device_score = device_vehicle_score_lookup.get(
                            (device_index, vehicle_index),
                            float("-inf")
                        )

                        if device_score == float("-inf"):
                            continue

                        combined_score = (
                            vehicle_score * device_score
                        )

                        if combined_score > best_score:

                            best_score = combined_score

                            best_candidate = (
                                vehicle_index,
                                incident_index,
                                radio,
                                device
                            )

        # Stop when no additional feasible assignment exists.
        if best_candidate is None:
            break

        vehicle_index, incident_index, radio, device = best_candidate

        vehicle_id = vehicle_ids[vehicle_index]
        incident_id = incident_ids[incident_index]

        # Assign the additional vehicle to the incident.
        selected_incidents[vehicle_index] = incident_index

        vehicles_per_incident[incident_id].append(
            vehicle_index
        )

        solution.vehicle_assignments.append(
            VehicleAssignment(
                vehicle_id=vehicle_id,
                incident_id=incident_id
            )
        )

        # Assign its mandatory Radio.
        radio_id = radio["device_id"]

        solution.device_assignments.append(
            DeviceAssignment(
                device_id=radio_id,
                vehicle_id=vehicle_id
            )
        )

        assigned_device_ids.add(radio_id)

        assigned_devices_by_vehicle[vehicle_id].append(
            radio_id
        )

        # Assign at least one currently required device.
        device_id = device["device_id"]

        solution.device_assignments.append(
            DeviceAssignment(
                device_id=device_id,
                vehicle_id=vehicle_id
            )
        )

        assigned_device_ids.add(device_id)

        assigned_devices_by_vehicle[vehicle_id].append(
            device_id
        )

    print("\nADDITIONAL VEHICLE ALLOCATION")
    print(
        "Total vehicles deployed:",
        len(solution.vehicle_assignments)
    )

    print(
        "Total devices assigned:",
        len(solution.device_assignments)
    )

    return solution