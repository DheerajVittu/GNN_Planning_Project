
import math


def calculate_objective(
    scenario,
    solution,
    distance_weight=1.0,
    operating_cost_weight=1.0
):
    """
    Calculate the combined planning objective.

    Objective:
        Total Cost = alpha * Total Distance
                   + beta * Total Hourly Operating Cost

    scenario: Dictionary loaded from the scenario JSON.
    solution: PlanningSolution object.
    """

    if distance_weight < 0 or operating_cost_weight < 0:
        raise ValueError("Objective weights cannot be negative.")

    # Create lookup dictionaries.
    vehicles = {
        vehicle["vehicle_id"]: vehicle
        for vehicle in scenario["vehicles"]
    }

    incidents = {
        incident["incident_id"]: incident
        for incident in scenario["incidents"]
    }

    devices = {
        device["device_id"]: device
        for device in scenario["devices"]
    }

    # -----------------------------------------
    # 1. Calculate total vehicle travel distance
    # -----------------------------------------

    total_distance = 0.0
    vehicle_distances = {}

    for assignment in solution.vehicle_assignments:

        vehicle_id = assignment.vehicle_id
        incident_id = assignment.incident_id

        if vehicle_id not in vehicles:
            raise ValueError(f"Unknown vehicle: {vehicle_id}")

        if incident_id not in incidents:
            raise ValueError(f"Unknown incident: {incident_id}")

        vehicle = vehicles[vehicle_id]
        incident = incidents[incident_id]

        distance = math.sqrt(
            (vehicle["x"] - incident["x"]) ** 2
            + (vehicle["y"] - incident["y"]) ** 2
            + (vehicle["z"] - incident["z"]) ** 2
        )

        vehicle_distances[vehicle_id] = distance
        total_distance += distance

    # -----------------------------------------
    # 2. Calculate hourly operating costs
    #    of deployed devices
    # -----------------------------------------

    total_operating_cost = 0.0
    deployed_devices = set()

    for assignment in solution.device_assignments:

        device_id = assignment.device_id

        if device_id not in devices:
            raise ValueError(f"Unknown device: {device_id}")

        if device_id in deployed_devices:
            raise ValueError(
                f"Device counted more than once: {device_id}"
            )

        deployed_devices.add(device_id)

        total_operating_cost += devices[device_id][
            "operating_cost"
        ]

    # -----------------------------------------
    # 3. Calculate combined objective
    # -----------------------------------------

    weighted_distance = distance_weight * total_distance

    weighted_operating_cost = (
        operating_cost_weight * total_operating_cost
    )

    total_cost = weighted_distance + weighted_operating_cost

    return {
        "total_distance": total_distance,
        "total_hourly_operating_cost": total_operating_cost,
        "distance_weight": distance_weight,
        "operating_cost_weight": operating_cost_weight,
        "weighted_distance": weighted_distance,
        "weighted_operating_cost": weighted_operating_cost,
        "total_objective": total_cost,
        "vehicle_distances": vehicle_distances,
        "deployed_device_count": len(deployed_devices)
    }
