
from collections import defaultdict


def _get_items(scenario, key):
    """Get a list of entities from the scenario dictionary."""
    return scenario.get(key, [])


def _get_solution_items(solution, key):
    """Support both dictionary-based and dataclass-based solutions."""
    if isinstance(solution, dict):
        return solution.get(key, [])
    return getattr(solution, key, [])


def _get_value(item, key):
    """Read a field from either a dictionary or a dataclass object."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _check_device_fit(device, vehicle):
    """
    Check whether one device fits inside a vehicle.

    Provisional assumption: rotation is allowed.
    This matches the current graph builder's dimension-sorting logic.
    """
    device_dimensions = sorted([
        device["length"],
        device["width"],
        device["height"]
    ])

    vehicle_dimensions = sorted([
        vehicle["loading_length"],
        vehicle["loading_width"],
        vehicle["loading_height"]
    ])

    return all(
        device_dimensions[i] <= vehicle_dimensions[i]
        for i in range(3)
    )


def validate_solution(scenario, solution):
    """
    Validate a planning solution against the six hard constraints.

    Expected scenario format:
        A dictionary containing incidents, devices, storages and vehicles.

    Expected solution:
        A PlanningSolution dataclass object or a dictionary.

    Returns:
        A dictionary containing the overall result, individual checks,
        and detailed violation messages.
    """

    incidents = _get_items(scenario, "incidents")
    devices = _get_items(scenario, "devices")
    storages = _get_items(scenario, "storages")
    vehicles = _get_items(scenario, "vehicles")

    device_assignments = _get_solution_items(
        solution, "device_assignments"
    )
    vehicle_assignments = _get_solution_items(
        solution, "vehicle_assignments"
    )

    # Create lookup dictionaries.
    incident_by_id = {
        item["incident_id"]: item for item in incidents
    }
    device_by_id = {
        item["device_id"]: item for item in devices
    }
    vehicle_by_id = {
        item["vehicle_id"]: item for item in vehicles
    }

    checks = {}
    violations = []

    def record_check(name, messages):
        checks[name] = {
            "passed": len(messages) == 0,
            "violations": messages
        }
        violations.extend(messages)

    # --------------------------------------------------
    # PRECHECK: Validate assignment references
    # --------------------------------------------------

    reference_errors = []

    for assignment in device_assignments:
        device_id = _get_value(assignment, "device_id")
        vehicle_id = _get_value(assignment, "vehicle_id")

        if device_id not in device_by_id:
            reference_errors.append(
                f"Unknown device ID: {device_id}"
            )

        if vehicle_id not in vehicle_by_id:
            reference_errors.append(
                f"Unknown vehicle ID: {vehicle_id}"
            )

    for assignment in vehicle_assignments:
        vehicle_id = _get_value(assignment, "vehicle_id")
        incident_id = _get_value(assignment, "incident_id")

        if vehicle_id not in vehicle_by_id:
            reference_errors.append(
                f"Unknown vehicle ID: {vehicle_id}"
            )

        if incident_id not in incident_by_id:
            reference_errors.append(
                f"Unknown incident ID: {incident_id}"
            )

    record_check("assignment_references", reference_errors)

    # Avoid unsafe lookups if references are invalid.
    valid_device_assignments = [
        a for a in device_assignments
        if _get_value(a, "device_id") in device_by_id
        and _get_value(a, "vehicle_id") in vehicle_by_id
    ]

    valid_vehicle_assignments = [
        a for a in vehicle_assignments
        if _get_value(a, "vehicle_id") in vehicle_by_id
        and _get_value(a, "incident_id") in incident_by_id
    ]

    # Group assigned devices by vehicle.
    devices_by_vehicle = defaultdict(list)

    for assignment in valid_device_assignments:
        device_id = _get_value(assignment, "device_id")
        vehicle_id = _get_value(assignment, "vehicle_id")

        devices_by_vehicle[vehicle_id].append(device_id)

    # Group assigned incidents by vehicle.
    incidents_by_vehicle = defaultdict(list)

    for assignment in valid_vehicle_assignments:
        vehicle_id = _get_value(assignment, "vehicle_id")
        incident_id = _get_value(assignment, "incident_id")

        incidents_by_vehicle[vehicle_id].append(incident_id)

    # --------------------------------------------------
    # CONSTRAINT 1:
    # Each individual device must fit inside its vehicle.
    # --------------------------------------------------

    fit_errors = []

    for assignment in valid_device_assignments:
        device_id = _get_value(assignment, "device_id")
        vehicle_id = _get_value(assignment, "vehicle_id")

        device = device_by_id[device_id]
        vehicle = vehicle_by_id[vehicle_id]

        if not _check_device_fit(device, vehicle):
            fit_errors.append(
                f"Device {device_id} does not fit inside "
                f"vehicle {vehicle_id}."
            )

    record_check("individual_device_fit", fit_errors)

    # --------------------------------------------------
    # CONSTRAINT 2:
    # Combined device dimensions must not exceed
    # the vehicle's loading dimensions.
    #
    # Simplified provisional approach:
    # Sort each device's dimensions before adding them.
    # --------------------------------------------------

    capacity_errors = []

    for vehicle_id, assigned_device_ids in devices_by_vehicle.items():
        vehicle = vehicle_by_id[vehicle_id]

        total_dimensions = [0, 0, 0]

        for device_id in assigned_device_ids:
            device = device_by_id[device_id]

            sorted_dimensions = sorted([
                device["length"],
                device["width"],
                device["height"]
            ])

            for i in range(3):
                total_dimensions[i] += sorted_dimensions[i]

        vehicle_dimensions = sorted([
            vehicle["loading_length"],
            vehicle["loading_width"],
            vehicle["loading_height"]
        ])

        if any(
            total_dimensions[i] > vehicle_dimensions[i]
            for i in range(3)
        ):
            capacity_errors.append(
                f"Vehicle {vehicle_id} exceeds its combined "
                f"loading dimensions. Device totals: "
                f"{total_dimensions}; vehicle capacity: "
                f"{vehicle_dimensions}."
            )

    record_check("combined_loading_dimensions", capacity_errors)

    # --------------------------------------------------
    # CONSTRAINT 3:
    # Every deployed vehicle must carry at least one radio.
    # --------------------------------------------------

    radio_errors = []

    for vehicle_id in incidents_by_vehicle:
        assigned_device_ids = devices_by_vehicle.get(
            vehicle_id, []
        )

        has_radio = any(
            device_by_id[device_id]["device_type"].lower() == "radio"
            for device_id in assigned_device_ids
        )

        if not has_radio:
            radio_errors.append(
                f"Deployed vehicle {vehicle_id} has no Radio device."
            )

    record_check("radio_per_deployed_vehicle", radio_errors)

    # --------------------------------------------------
    # CONSTRAINT 4:
    # Each device can be assigned to only one vehicle.
    # --------------------------------------------------

    device_errors = []
    assigned_device_ids = [
        _get_value(a, "device_id")
        for a in valid_device_assignments
    ]

    seen_devices = set()

    for device_id in assigned_device_ids:
        if device_id in seen_devices:
            device_errors.append(
                f"Device {device_id} is assigned more than once."
            )
        seen_devices.add(device_id)

    record_check("device_assigned_once", device_errors)

    # --------------------------------------------------
    # CONSTRAINT 5:
    # Each vehicle can serve only one incident.
    # --------------------------------------------------

    vehicle_errors = []

    for vehicle_id, assigned_incident_ids in incidents_by_vehicle.items():
        if len(set(assigned_incident_ids)) > 1:
            vehicle_errors.append(
                f"Vehicle {vehicle_id} is assigned to multiple incidents: "
                f"{sorted(set(assigned_incident_ids))}."
            )

        if len(assigned_incident_ids) > 1:
            vehicle_errors.append(
                f"Vehicle {vehicle_id} has duplicate incident assignments."
            )

    record_check("vehicle_serves_one_incident", vehicle_errors)

    # --------------------------------------------------
    # CONSTRAINT 6:
    # Assigned devices must come from available storage stock.
    # --------------------------------------------------

    storage_errors = []
    device_storage_count = defaultdict(int)
    devices_in_storage = defaultdict(set)

    for storage in storages:
        storage_id = storage["storage_id"]

        for device_id in storage.get("device_ids", []):
            device_storage_count[device_id] += 1
            devices_in_storage[storage_id].add(device_id)

    for device_id, count in device_storage_count.items():
        if count > 1:
            storage_errors.append(
                f"Device {device_id} appears in more than one storage."
            )

    for assignment in valid_device_assignments:
        device_id = _get_value(assignment, "device_id")

        if device_storage_count[device_id] == 0:
            storage_errors.append(
                f"Device {device_id} is assigned but is not available "
                f"in any storage."
            )

    record_check("storage_stock_availability", storage_errors)

    # --------------------------------------------------
    # ADDITIONAL CHECK:
    # Devices assigned to a vehicle must match the demand
    # of the incident served by that vehicle.
    #
    # This is reported separately from the six constraints.
    # --------------------------------------------------

    
    # --------------------------------------------------
    # ADDITIONAL CHECK:
    # Aggregate all device assignments across vehicles
    # serving the same incident.
    # --------------------------------------------------

    demand_errors = []
    demand_summary = {}

    
    demand_summary = {}

    assigned_counts_by_incident = defaultdict(
        lambda: defaultdict(int)
    )

    # Collect devices from every vehicle serving an incident.
    for vehicle_id, assigned_incident_ids in incidents_by_vehicle.items():

        if len(set(assigned_incident_ids)) != 1:
            continue

        incident_id = assigned_incident_ids[0]

        for device_id in devices_by_vehicle.get(vehicle_id, []):

            device_type = device_by_id[device_id]["device_type"]

            # Radio is mandatory vehicle equipment,
            # not an incident delivery.
            if device_type.lower() == "radio":
                continue

            assigned_counts_by_incident[incident_id][device_type] += 1

    # Overall coverage statistics.
    total_required = 0
    total_fulfilled = 0
    incidents_receiving_devices = 0

    # Evaluate every incident, including incidents receiving no devices.
    for incident_id, incident in incident_by_id.items():

        required_devices = incident.get("required_devices", {})
        assigned_counts = assigned_counts_by_incident[incident_id]

        fulfilled = {}
        incident_required = 0
        incident_fulfilled = 0

        for device_type, required_count in required_devices.items():

            assigned_count = assigned_counts.get(device_type, 0)

            fulfilled[device_type] = min(
                assigned_count,
                required_count
            )

            # Exclude Radio from delivery coverage.
            if device_type.lower() == "radio":
                continue

            incident_required += required_count
            incident_fulfilled += fulfilled[device_type]

            if assigned_count > required_count:
                demand_errors.append(
                    f"Incident {incident_id} receives "
                    f"{assigned_count} {device_type} device(s), "
                    f"but only {required_count} are required."
                )

        incident_unmet = incident_required - incident_fulfilled

        fulfillment_percentage = (
            incident_fulfilled / incident_required * 100
            if incident_required > 0 else 100.0
        )

        if incident_fulfilled > 0:
            incidents_receiving_devices += 1

        total_required += incident_required
        total_fulfilled += incident_fulfilled

        demand_summary[incident_id] = {
            "required": required_devices,
            "assigned": dict(assigned_counts),
            "fulfilled": fulfilled,
            "total_required_excluding_radio": incident_required,
            "total_fulfilled_excluding_radio": incident_fulfilled,
            "unmet_excluding_radio": incident_unmet,
            "fulfillment_percentage": round(
                fulfillment_percentage, 2
            )
        }

    overall_coverage = {
        "total_incidents": len(incident_by_id),
        "incidents_receiving_devices": incidents_receiving_devices,
        "incidents_without_devices": (
            len(incident_by_id) - incidents_receiving_devices
        ),
        "incident_coverage_percentage": round(
            incidents_receiving_devices / len(incident_by_id) * 100,
            2
        ) if incident_by_id else 0.0,
        "total_required_excluding_radio": total_required,
        "total_fulfilled_excluding_radio": total_fulfilled,
        "total_unmet_excluding_radio": total_required - total_fulfilled,
        "overall_fulfillment_percentage": round(
            total_fulfilled / total_required * 100,
            2
        ) if total_required else 100.0
    }

    record_check("incident_demand_matching", demand_errors)


    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    # The six hard constraints and reference precheck determine
    # hard-constraint validity. Demand matching is reported separately.
    hard_constraint_names = [
        "assignment_references",
        "individual_device_fit",
        "combined_loading_dimensions",
        "radio_per_deployed_vehicle",
        "device_assigned_once",
        "vehicle_serves_one_incident",
        "storage_stock_availability"
    ]

    hard_constraints_passed = all(
        checks[name]["passed"]
        for name in hard_constraint_names
    )

    return {
        "valid": hard_constraints_passed,
        "hard_constraints_passed": hard_constraints_passed,
        "checks": checks,
        "violations": violations,
        "demand_summary": demand_summary,
        "overall_coverage": overall_coverage
    }
