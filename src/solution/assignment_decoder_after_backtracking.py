
import torch
from collections import defaultdict

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)

from src.solution.objective_function import calculate_objective


def _vehicle_incident_distance(vehicle, incident):
    """
    Calculate straight-line 3D distance
    between a vehicle and an incident.
    """
    return (
        (vehicle["x"] - incident["x"]) ** 2
        + (vehicle["y"] - incident["y"]) ** 2
        + (vehicle["z"] - incident["z"]) ** 2
    ) ** 0.5

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

        # Skip incidents that require only Radios.
        incident = incident_by_id[incident_id]

        has_non_radio_demand = any(
            device_type.lower() != "radio" and quantity > 0
            for device_type, quantity
            in incident["required_devices"].items()
        )

        if not has_non_radio_demand:
            continue

        
        # Combine GNN confidence with travel-distance desirability.
        candidates = []

        # Find the maximum possible distance for this incident.
        incident_distances = []
        feasible_vehicle_indices = set()
        for _, vehicle_index, candidate_incident_index in vehicle_incident_candidates:

            if candidate_incident_index != incident_index:
                continue

            vehicle = vehicle_by_id[vehicle_ids[vehicle_index]]


            # Check whether this vehicle can carry at least
            # one available Radio and one required device.

            has_feasible_equipment = False

            for radio in devices:

                radio_id = radio["device_id"]

                if radio["device_type"].lower() != "radio":
                    continue

                if radio_id not in available_device_ids:
                    continue

                if not _can_add_device(
                    radio,
                    vehicle,
                    [],
                    device_by_id
                ):
                    continue

                try:
                    radio_index = device_ids.index(radio_id)
                except ValueError:
                    continue

                radio_score = device_vehicle_score_lookup.get(
                    (radio_index, vehicle_index),
                    float("-inf")
                )

                if radio_score == float("-inf"):
                    continue

                for device_type, quantity in incident["required_devices"].items():

                    if device_type.lower() == "radio" or quantity <= 0:
                        continue

                    for required_device in devices:

                        required_id = required_device["device_id"]

                        if required_device["device_type"] != device_type:
                            continue

                        if required_id not in available_device_ids:
                            continue

                        if required_id == radio_id:
                            continue

                        if not _can_add_device(
                            required_device,
                            vehicle,
                            [radio_id],
                            device_by_id
                        ):
                            continue

                        try:
                            required_index = device_ids.index(required_id)
                        except ValueError:
                            continue

                        required_score = device_vehicle_score_lookup.get(
                            (required_index, vehicle_index),
                            float("-inf")
                        )

                        if required_score == float("-inf"):
                            continue

                        has_feasible_equipment = True
                        break

                    if has_feasible_equipment:
                        break

                if has_feasible_equipment:
                    break

            if not has_feasible_equipment:
                continue
            feasible_vehicle_indices.add(vehicle_index)
            distance = (
                (vehicle["x"] - incident["x"]) ** 2
                + (vehicle["y"] - incident["y"]) ** 2
                + (vehicle["z"] - incident["z"]) ** 2
            ) ** 0.5

            incident_distances.append(distance)

        max_distance = max(incident_distances, default=1.0)

        for gnn_score, vehicle_index, candidate_incident_index in vehicle_incident_candidates:

            if candidate_incident_index != incident_index:
                continue

            if vehicle_index not in available_vehicle_indices:
                continue
            if vehicle_index not in feasible_vehicle_indices:
                continue
            vehicle = vehicle_by_id[vehicle_ids[vehicle_index]]

            distance = (
                (vehicle["x"] - incident["x"]) ** 2
                + (vehicle["y"] - incident["y"]) ** 2
                + (vehicle["z"] - incident["z"]) ** 2
            ) ** 0.5

            distance_score = (
                1.0 - distance / max_distance
                if max_distance > 0 else 1.0
            )

            # Estimate how many required devices this vehicle
            # can carry together with one available Radio.
            best_device_count = 0

            for radio in devices:

                radio_id = radio["device_id"]

                if radio["device_type"].lower() != "radio":
                    continue

                if radio_id not in available_device_ids:
                    continue

                try:
                    radio_index = device_ids.index(radio_id)
                except ValueError:
                    continue

                if (radio_index, vehicle_index) not in device_vehicle_score_lookup:
                    continue

                if not _can_add_device(radio, vehicle, [], device_by_id):
                    continue

                # Prepare available candidates for every required device type.
                required_type_candidates = []

                for device_type, quantity in incident["required_devices"].items():

                    if device_type.lower() == "radio" or quantity <= 0:
                        continue

                    candidates_for_type = []

                    for required_device in devices:

                        required_id = required_device["device_id"]

                        if required_device["device_type"].lower() != device_type.lower():
                            continue

                        if required_id not in available_device_ids:
                            continue

                        try:
                            required_index = device_ids.index(required_id)
                        except ValueError:
                            continue

                        if (required_index, vehicle_index) not in device_vehicle_score_lookup:
                            continue

                        candidates_for_type.append(required_device)

                    required_type_candidates.append(
                        (quantity, candidates_for_type)
                    )

                # Explore alternative combinations of required devices.
                best_for_radio = [0]

                def search_types(type_position, current_ids, fulfilled_count):

                    if type_position >= len(required_type_candidates):
                        best_for_radio[0] = max(
                            best_for_radio[0],
                            fulfilled_count
                        )
                        return

                    quantity, candidates = required_type_candidates[type_position]

                    def choose_devices(candidate_position, selected_count, selected_ids):

                        if selected_count >= quantity or candidate_position >= len(candidates):
                            search_types(
                                type_position + 1,
                                selected_ids,
                                fulfilled_count + selected_count
                            )
                            return

                        # Option 1: Skip this device.
                        choose_devices(
                            candidate_position + 1,
                            selected_count,
                            selected_ids
                        )

                        # Option 2: Include this device if capacity allows.
                        candidate = candidates[candidate_position]
                        candidate_id = candidate["device_id"]

                        if _can_add_device(
                            candidate,
                            vehicle,
                            selected_ids,
                            device_by_id
                        ):
                            choose_devices(
                                candidate_position + 1,
                                selected_count + 1,
                                selected_ids + [candidate_id]
                            )

                    choose_devices(0, 0, current_ids)

                search_types(0, [radio_id], 0)

                best_device_count = max(
                    best_device_count,
                    best_for_radio[0]
                )

            total_required_devices = sum(
                quantity
                for device_type, quantity in incident["required_devices"].items()
                if device_type.lower() != "radio" and quantity > 0
            )

            equipment_score = (
                best_device_count / total_required_devices
                if total_required_devices > 0 else 0.0
            )

            # Equal initial weights; configurable for later experiments.
            gnn_weight = 1.0 / 3.0
            distance_weight = 1.0 / 3.0
            equipment_weight = 1.0 / 3.0

            combined_score = (
                gnn_weight * gnn_score
                + distance_weight * distance_score
                + equipment_weight * equipment_score
            )

            if "S47" in scenario["scenario_id"] and incident_id == "Incident_3":
                print(
                    f"[S47 DEBUG] Vehicle: {vehicle_ids[vehicle_index]} | "
                    f"GNN: {gnn_score:.4f} | "
                    f"Distance: {distance:.2f} | "
                    f"Distance score: {distance_score:.4f} | "
                    f"Equipment: {equipment_score:.4f} | "
                    f"Combined: {combined_score:.4f}"
                )

            candidates.append(
                (combined_score, vehicle_index, gnn_score, distance)
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        if not candidates:
            continue

        _, selected_vehicle_index, _, _ = candidates[0]

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
    # Prioritize vehicles using the score for their actual
    # selected vehicle-incident assignment.
    selected_incidents_ordered = sorted(
        deployed_vehicle_indices,
        key=lambda index: next(
            (
                score
                for score, v_index, i_index in vehicle_incident_candidates
                if v_index == index
                and i_index == selected_incidents[index]
            ),
            float("-inf")
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
            unique_candidate_devices = {
                item[1] for item in candidates
            }

            print(
                f"[STEP 6] Incident={incident_id}, "
                f"Type={device_type}, "
                f"Required={required_quantity}, "
                f"Unique candidate devices={len(unique_candidate_devices)}, "
                f"Candidate pairs={len(candidates)}"
            )
            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            allocated = 0
            capacity_rejections = 0
            already_assigned_skips = 0
            for score, device_id, vehicle_id in candidates:

                if allocated >= required_quantity:
                    break

                if device_id in assigned_device_ids:
                    already_assigned_skips += 1
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
                    capacity_rejections += 1
                    print(
                        f"[STEP 6 REJECTED] Device={device_id}, "
                        f"Vehicle={vehicle_id}, "
                        f"Type={device_type}"
                    )
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

            print(
                f"[STEP 6 DIAGNOSIS] {incident_id} | {device_type} | "
                f"Required={required_quantity} | "
                f"Allocated={allocated} | "
                f"Capacity rejections={capacity_rejections} | "
                f"Already assigned skips={already_assigned_skips}"
            )
            print(
                f"[STEP 6] Incident={incident_id}, "
                f"Type={device_type}, "
                f"Successfully allocated={allocated}/{required_quantity}"
            )

    # TEMPORARY DIAGNOSTIC: fulfilment after Step 6
    print("\nDEVICE FULFILMENT AFTER STEP 6")

    for incident_id in incident_ids:

        counts = defaultdict(int)

        for vehicle_assignment in solution.vehicle_assignments:

            if vehicle_assignment.incident_id != incident_id:
                continue

            for device_assignment in solution.device_assignments:

                if device_assignment.vehicle_id != vehicle_assignment.vehicle_id:
                    continue

                device_type = device_by_id[
                    device_assignment.device_id
                ]["device_type"]

                if device_type.lower() != "radio":
                    counts[device_type] += 1

        print(incident_id, dict(counts))

    # --------------------------------------------------
    # STEP 7: DEPLOY ADDITIONAL VEHICLES WHEN NEEDED
    # --------------------------------------------------

    # The first allocation pass may leave some incident
    # requirements unfulfilled. Try assigning additional
    # vehicles to those incidents.

    while True:

        feasible_candidates = []

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

                        distance = _vehicle_incident_distance(
                            vehicle,
                            incident
                        )

                        incident_distances = [
                            _vehicle_incident_distance(
                                vehicle_by_id[vehicle_ids[v_index]],
                                incident
                            )
                            for _, v_index, i_index in vehicle_incident_candidates
                            if i_index == incident_index
                        ]

                        max_distance = max(incident_distances, default=1.0)

                        distance_score = (
                            1.0 - distance / max_distance
                            if max_distance > 0 else 1.0
                        )

                        gnn_combined_score = vehicle_score * device_score

                        alpha = 0.5

                        combined_score = (
                            alpha * gnn_combined_score
                            + (1.0 - alpha) * distance_score
                        )
                        feasible_candidates.append((
                            combined_score,
                            vehicle_index,
                            incident_index,
                            radio,
                            device
                        ))

        # Stop when no additional feasible assignment exists.
        
        # --------------------------------------------------
        # LOOK-AHEAD: SELECT THE BEST COMPATIBLE PAIR
        # --------------------------------------------------

        best_candidate = None
        best_total_score = float("-inf")

        # Count devices already assigned to each incident.
        assigned_counts = defaultdict(int)

        for assignment in solution.device_assignments:

            assigned_incident_index = selected_incidents.get(
                vehicle_ids.index(assignment.vehicle_id)
            )

            if assigned_incident_index is not None:
                assigned_device = device_by_id[assignment.device_id]

                assigned_counts[
                    (
                        assigned_incident_index,
                        assigned_device["device_type"]
                    )
                ] += 1

        # Evaluate each candidate together with a possible
        # second candidate.
        for first_index, first in enumerate(feasible_candidates):

            (
                first_score,
                first_vehicle,
                first_incident,
                first_radio,
                first_device
            ) = first

            pair_score = first_score

            for second_index, second in enumerate(feasible_candidates):

                if second_index == first_index:
                    continue

                (
                    second_score,
                    second_vehicle,
                    second_incident,
                    second_radio,
                    second_device
                ) = second

                # Two assignments cannot use the same vehicle.
                if first_vehicle == second_vehicle:
                    continue

                # They cannot reuse the same Radio or device.
                if first_radio["device_id"] == second_radio["device_id"]:
                    continue

                if first_device["device_id"] == second_device["device_id"]:
                    continue

                # If both serve the same incident and require
                # the same device type, check that two units
                # are actually still needed.
                if (
                    first_incident == second_incident
                    and first_device["device_type"]
                    == second_device["device_type"]
                ):
                    incident = incident_by_id[
                        incident_ids[first_incident]
                    ]

                    device_type = first_device["device_type"]

                    remaining = (
                        incident["required_devices"].get(device_type, 0)
                        - assigned_counts.get(
                            (first_incident, device_type),
                            0
                        )
                    )

                    if remaining < 2:
                        continue

                # Prefer the first candidate that gives the
                # highest combined score with a compatible second.
                pair_score = max(
                    pair_score,
                    first_score + second_score
                )

            if pair_score > best_total_score:
                best_total_score = pair_score
                best_candidate = first

        # Stop when no feasible assignment exists.
        if best_candidate is None:
            break

        (
            _,
            vehicle_index,
            incident_index,
            radio,
            device
        ) = best_candidate


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


    # TEMPORARY DIAGNOSTIC: device fulfilment before Step 8
    print("\nDEVICE FULFILMENT BEFORE STEP 8")

    for incident_id in incident_ids:

        counts = defaultdict(int)

        for vehicle_assignment in solution.vehicle_assignments:

            if vehicle_assignment.incident_id != incident_id:
                continue

            for device_assignment in solution.device_assignments:

                if device_assignment.vehicle_id != vehicle_assignment.vehicle_id:
                    continue

                device_type = device_by_id[
                    device_assignment.device_id
                ]["device_type"]

                if device_type.lower() != "radio":
                    counts[device_type] += 1

        print(incident_id, dict(counts))

    
    # --------------------------------------------------
    # STEP 8: CONSOLIDATE VEHICLES SERVING THE SAME INCIDENT
    # --------------------------------------------------

    for incident_id, incident_vehicle_indices in list(
        vehicles_per_incident.items()
    ):

        active_indices = [
            index
            for index in incident_vehicle_indices
            if index in selected_incidents
        ]

        if len(active_indices) <= 1:
            continue

        incident_vehicle_ids = {
            vehicle_ids[index]
            for index in active_indices
        }

        # Collect all non-Radio devices assigned
        # to vehicles serving this incident.
        incident_device_ids = [
            assignment.device_id
            for assignment in solution.device_assignments
            if assignment.vehicle_id in incident_vehicle_ids
            and device_by_id[
                assignment.device_id
            ]["device_type"].lower() != "radio"
        ]

        if not incident_device_ids:
            continue


        feasible_targets = []

        # Test whether one vehicle can carry all devices.
        for target_index in active_indices:

            target_vehicle_id = vehicle_ids[target_index]
            target_vehicle = vehicle_by_id[target_vehicle_id]

            target_radios = [
                assignment.device_id
                for assignment in solution.device_assignments
                if assignment.vehicle_id == target_vehicle_id
                and device_by_id[
                    assignment.device_id
                ]["device_type"].lower() == "radio"
            ]

            if len(target_radios) != 1:
                continue

            # Start with the target vehicle's Radio.
            test_device_ids = list(target_radios)
            total_score = 0.0
            feasible = True

            for device_id in incident_device_ids:

                device = device_by_id[device_id]

                try:
                    device_index = device_ids.index(device_id)
                except ValueError:
                    feasible = False
                    break

                device_score = device_vehicle_score_lookup.get(
                    (device_index, target_index),
                    float("-inf")
                )

                # A valid candidate graph edge is required.
                if device_score == float("-inf"):
                    feasible = False
                    break

                # Check individual and combined capacity.
                if not _can_add_device(
                    device,
                    target_vehicle,
                    test_device_ids,
                    device_by_id
                ):
                    feasible = False
                    break

                test_device_ids.append(device_id)
                total_score += device_score

            if not feasible:
                continue

            # Build the resulting device assignments.
            candidate_device_assignments = []

            for assignment in solution.device_assignments:

                if assignment.vehicle_id not in incident_vehicle_ids:
                    candidate_device_assignments.append(assignment)
                    continue

                device = device_by_id[assignment.device_id]

                if device["device_type"].lower() == "radio":

                    if assignment.vehicle_id == target_vehicle_id:
                        candidate_device_assignments.append(assignment)

                else:
                    candidate_device_assignments.append(
                        DeviceAssignment(
                            device_id=assignment.device_id,
                            vehicle_id=target_vehicle_id
                        )
                    )

            # Build the resulting vehicle assignments.
            candidate_vehicle_assignments = [
                assignment
                for assignment in solution.vehicle_assignments
                if assignment.vehicle_id not in incident_vehicle_ids
                or assignment.vehicle_id == target_vehicle_id
            ]

            # Create a temporary candidate solution.
            candidate_solution = PlanningSolution(
                solution_id=solution.solution_id,
                scenario_id=solution.scenario_id,
                device_assignments=candidate_device_assignments,
                vehicle_assignments=candidate_vehicle_assignments
            )

            # Calculate the actual planning objective.
            objective_result = calculate_objective(
                scenario,
                candidate_solution,
                distance_weight=1.0,
                operating_cost_weight=1.0
            )

            total_objective = objective_result["total_objective"]

            # Store objective, GNN score, target and devices.
            feasible_targets.append(
                (
                    total_objective,
                    -total_score,
                    target_index,
                    test_device_ids
                )
            )

        if not feasible_targets:
            continue

        # Choose the lowest objective.
        # Use GNN score only to break ties.
        feasible_targets.sort(
            key=lambda item: (item[0], item[1])
        )

        (
            best_objective,
            _,
            target_index,
            consolidated_device_ids
        ) = feasible_targets[0]

        target_vehicle_id = vehicle_ids[target_index]

        print(
            "\nCONSOLIDATING VEHICLES FOR",
            incident_id,
            "ONTO",
            target_vehicle_id,
            "| Objective:",
            round(best_objective, 2)
        )

        # Keep devices on one vehicle and remove redundant Radios.
        updated_device_assignments = []

        for assignment in solution.device_assignments:

            if assignment.vehicle_id not in incident_vehicle_ids:
                updated_device_assignments.append(assignment)
                continue

            device = device_by_id[assignment.device_id]

            if device["device_type"].lower() == "radio":

                if assignment.vehicle_id == target_vehicle_id:
                    updated_device_assignments.append(assignment)

            else:

                updated_device_assignments.append(
                    DeviceAssignment(
                        device_id=assignment.device_id,
                        vehicle_id=target_vehicle_id
                    )
                )

        solution.device_assignments = updated_device_assignments

        # Remove redundant vehicle assignments.
        solution.vehicle_assignments = [
            assignment
            for assignment in solution.vehicle_assignments
            if assignment.vehicle_id not in incident_vehicle_ids
            or assignment.vehicle_id == target_vehicle_id
        ]

        # Update decoder tracking structures.
        for index in active_indices:

            if index != target_index:
                selected_incidents.pop(index, None)

                assigned_devices_by_vehicle[
                    vehicle_ids[index]
                ] = []

        vehicles_per_incident[incident_id] = [target_index]

        assigned_devices_by_vehicle[target_vehicle_id] = (
            consolidated_device_ids
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