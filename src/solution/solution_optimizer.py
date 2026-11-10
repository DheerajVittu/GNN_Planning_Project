from collections import defaultdict

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)

from src.solution.constraint_validator import validate_solution
from src.solution.objective_function import calculate_objective


def evaluate_solution(
    scenario,
    solution,
    distance_weight=1.0,
    operating_cost_weight=1.0
):
    """
    Evaluate a planning solution using:

    1. Hard-constraint validity
    2. Number of incidents receiving devices
    3. Total fulfilled demand
    4. Combined planning objective

    Higher coverage and fulfilment are preferred.
    Lower objective cost is preferred when the
    first two criteria are equal.
    """

    validation = validate_solution(scenario, solution)

    if not validation["hard_constraints_passed"]:
        return {
            "feasible": False,
            "ranking_score": None,
            "validation": validation
        }

    coverage = validation["overall_coverage"]

    objective = calculate_objective(
        scenario=scenario,
        solution=solution,
        distance_weight=distance_weight,
        operating_cost_weight=operating_cost_weight
    )

    incidents_served = coverage[
        "incidents_receiving_devices"
    ]

    fulfilled_devices = coverage[
        "total_fulfilled_excluding_radio"
    ]

    total_objective = objective["total_objective"]

    # Lexicographic ranking:
    # 1. Maximize incidents served.
    # 2. Maximize fulfilled devices.
    # 3. Minimize combined objective.

    ranking_score = (
        incidents_served,
        fulfilled_devices,
        -total_objective
    )

    return {
        "feasible": True,
        "incidents_served": incidents_served,
        "fulfilled_devices": fulfilled_devices,
        "fulfillment_percentage": coverage[
            "overall_fulfillment_percentage"
        ],
        "incident_coverage_percentage": coverage[
            "incident_coverage_percentage"
        ],
        "total_distance": objective["total_distance"],
        "hourly_operating_cost": objective[
            "total_hourly_operating_cost"
        ],
        "total_objective": total_objective,
        "ranking_score": ranking_score,
        "validation": validation
    }


def select_best_solution(
    scenario,
    candidate_solutions,
    distance_weight=1.0,
    operating_cost_weight=1.0
):
    """
    Evaluate multiple candidate solutions and return
    the best feasible solution.

    Priority:
    1. Maximize incidents receiving devices.
    2. Maximize fulfilled demand.
    3. Minimize combined objective.
    """

    best_solution = None
    best_evaluation = None

    evaluated_candidates = []

    for candidate in candidate_solutions:

        evaluation = evaluate_solution(
            scenario=scenario,
            solution=candidate,
            distance_weight=distance_weight,
            operating_cost_weight=operating_cost_weight
        )

        evaluated_candidates.append({
            "solution_id": candidate.solution_id,
            "feasible": evaluation["feasible"],
            "evaluation": evaluation
        })

        if not evaluation["feasible"]:
            continue

        if (
            best_evaluation is None
            or evaluation["ranking_score"]
            > best_evaluation["ranking_score"]
        ):
            best_solution = candidate
            best_evaluation = evaluation

    return {
        "best_solution": best_solution,
        "best_evaluation": best_evaluation,
        "evaluated_candidates": evaluated_candidates
    }


def generate_candidate_solutions(
    scenario,
    base_solution,
    graph,
    max_candidates=100
):
    """
    Generate bounded one-change alternatives from a base solution.

    Possible changes:
    1. Add an available required device.
    2. Replace an assigned non-Radio device with an
       available device needed by the same incident.

    Only candidates satisfying all hard constraints are returned.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")

    devices = {
        device["device_id"]: device
        for device in scenario["devices"]
    }

    incidents = {
        incident["incident_id"]: incident
        for incident in scenario["incidents"]
    }

    # Available inventory.
    available_device_ids = {
        device_id
        for storage in scenario["storages"]
        for device_id in storage.get("device_ids", [])
    }

    # Devices already assigned in the base solution.
    assigned_device_ids = {
        assignment.device_id
        for assignment in base_solution.device_assignments
    }

    # Vehicle-to-incident mapping.
    incident_by_vehicle = {
        assignment.vehicle_id: assignment.incident_id
        for assignment in base_solution.vehicle_assignments
    }

    # Build device-to-vehicle eligibility lookup.
    edge_index = graph[
        "device", "can_be_transported_by", "vehicle"
    ].edge_index.detach().cpu()

    device_node_ids = graph["device"].node_ids
    vehicle_node_ids = graph["vehicle"].node_ids

    eligible_pairs = set()

    for position in range(edge_index.shape[1]):
        device_index = int(edge_index[0, position])
        vehicle_index = int(edge_index[1, position])

        eligible_pairs.add((
            device_node_ids[device_index],
            vehicle_node_ids[vehicle_index]
        ))

    candidates = []
    seen_assignments = set()

    def add_candidate(device_assignments, change_name):
        """Create and validate a candidate solution."""

        if len(candidates) >= max_candidates:
            return

        assignment_key = tuple(sorted(
            (item.device_id, item.vehicle_id)
            for item in device_assignments
        ))

        if assignment_key in seen_assignments:
            return

        seen_assignments.add(assignment_key)

        candidate = PlanningSolution(
            solution_id=(
                f"{base_solution.solution_id}_"
                f"Candidate_{len(candidates) + 1}"
            ),
            scenario_id=base_solution.scenario_id,
            vehicle_assignments=[
                VehicleAssignment(
                    item.vehicle_id,
                    item.incident_id
                )
                for item in base_solution.vehicle_assignments
            ],
            device_assignments=[
                DeviceAssignment(
                    item.device_id,
                    item.vehicle_id
                )
                for item in device_assignments
            ]
        )

        validation = validate_solution(scenario, candidate)

        if validation["hard_constraints_passed"]:
            candidates.append(candidate)

    # Current device counts delivered to each incident.
    assigned_counts = defaultdict(lambda: defaultdict(int))

    for assignment in base_solution.device_assignments:

        device = devices[assignment.device_id]

        if device["device_type"].lower() == "radio":
            continue

        incident_id = incident_by_vehicle.get(
            assignment.vehicle_id
        )

        if incident_id is not None:
            assigned_counts[incident_id][
                device["device_type"]
            ] += 1

    # --------------------------------------------------
    # GENERATE ADD AND REPLACEMENT CANDIDATES
    # --------------------------------------------------

    for vehicle_id, incident_id in incident_by_vehicle.items():

        incident = incidents[incident_id]
        required_devices = incident["required_devices"]

        current_devices = [
            assignment
            for assignment in base_solution.device_assignments
            if assignment.vehicle_id == vehicle_id
            and devices[assignment.device_id][
                "device_type"
            ].lower() != "radio"
        ]

        for device_id in sorted(available_device_ids):

            if device_id in assigned_device_ids:
                continue

            device = devices[device_id]
            device_type = device["device_type"]

            # Radio is handled separately as vehicle equipment.
            if device_type.lower() == "radio":
                continue

            required_quantity = required_devices.get(
                device_type, 0
            )

            if required_quantity <= 0:
                continue

            current_quantity = assigned_counts[
                incident_id
            ][device_type]

            # --------------------------------------------------
            # OPTION A: ADD AN UNUSED REQUIRED DEVICE
            # --------------------------------------------------

            if current_quantity < required_quantity:

                if (device_id, vehicle_id) in eligible_pairs:

                    new_assignments = (
                        list(base_solution.device_assignments)
                        + [
                            DeviceAssignment(
                                device_id,
                                vehicle_id
                            )
                        ]
                    )

                    add_candidate(
                        new_assignments,
                        "add"
                    )

            # --------------------------------------------------
            # OPTION B: REPLACE AN ASSIGNED DEVICE
            # --------------------------------------------------

            for old_assignment in current_devices:

                old_device_type = devices[
                    old_assignment.device_id
                ]["device_type"]

                remaining_quantity = (
                    current_quantity
                    - (
                        1 if old_device_type == device_type
                        else 0
                    )
                )

                if remaining_quantity >= required_quantity:
                    continue

                if (device_id, vehicle_id) not in eligible_pairs:
                    continue

                new_assignments = [
                    assignment
                    for assignment in base_solution.device_assignments
                    if not (
                        assignment.device_id
                        == old_assignment.device_id
                        and assignment.vehicle_id == vehicle_id
                    )
                ]

                new_assignments.append(
                    DeviceAssignment(
                        device_id,
                        vehicle_id
                    )
                )

                add_candidate(
                    new_assignments,
                    "replace"
                )

            if len(candidates) >= max_candidates:
                return candidates

    return candidates

