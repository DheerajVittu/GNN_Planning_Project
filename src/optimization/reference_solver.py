
import json
import math

from ortools.sat.python import cp_model

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)

from src.solution.constraint_validator import validate_solution

from src.solution.objective_function import calculate_objective


def load_scenario(scenario_path):
    """
    Load a disaster-response scenario from a JSON file.
    """

    with open(scenario_path, "r", encoding="utf-8") as file:
        scenario = json.load(file)

    required_sections = [
        "incidents",
        "devices",
        "storages",
        "vehicles"
    ]

    for section in required_sections:
        if section not in scenario:
            raise ValueError(
                f"Scenario is missing required section: {section}"
            )

    return scenario


def build_reference_model(scenario):
    """
    Create the CP-SAT model and assignment variables.
    """

    model = cp_model.CpModel()

    devices = scenario["devices"]
    vehicles = scenario["vehicles"]
    incidents = scenario["incidents"]

    # Device -> Vehicle assignment variables
    x = {}

    for device in devices:
        device_id = device["device_id"]

        for vehicle in vehicles:
            vehicle_id = vehicle["vehicle_id"]

            x[(device_id, vehicle_id)] = model.NewBoolVar(
                f"device_{device_id}_vehicle_{vehicle_id}"
            )

    # Vehicle -> Incident assignment variables
    y = {}

    for vehicle in vehicles:
        vehicle_id = vehicle["vehicle_id"]

        for incident in incidents:
            incident_id = incident["incident_id"]

            y[(vehicle_id, incident_id)] = model.NewBoolVar(
                f"vehicle_{vehicle_id}_incident_{incident_id}"
            )

    return model, x, y


def add_device_uniqueness_constraint(model, x, scenario):
    """
    Ensure each device is assigned to at most one vehicle.
    """

    for device in scenario["devices"]:

        device_id = device["device_id"]

        model.Add(
            sum(
                x[(device_id, vehicle["vehicle_id"])]
                for vehicle in scenario["vehicles"]
            ) <= 1
        )


def add_storage_inventory_constraint(model, x, scenario):
    """
    Ensure that devices can only be assigned if they
    are available in exactly one storage.
    """

    device_storage_count = {
        device["device_id"]: 0
        for device in scenario["devices"]
    }

    # Count how many storages contain each device.
    for storage in scenario["storages"]:

        for device_id in storage["device_ids"]:

            if device_id not in device_storage_count:
                raise ValueError(
                    f"Storage contains unknown device: {device_id}"
                )

            device_storage_count[device_id] += 1

    # Validate that every device belongs to exactly one storage.
    for device_id, count in device_storage_count.items():

        if count == 0:
            raise ValueError(
                f"Device {device_id} is not available in any storage."
            )

        if count > 1:
            raise ValueError(
                f"Device {device_id} appears in multiple storages."
            )

    # Only devices available in storage can be assigned.
    # The validation above confirms that every device
    # belongs to exactly one storage.
    available_devices = set(device_storage_count.keys())

    for device in scenario["devices"]:

        device_id = device["device_id"]

        if device_id not in available_devices:

            for vehicle in scenario["vehicles"]:

                vehicle_id = vehicle["vehicle_id"]

                model.Add(
                    x[(device_id, vehicle_id)] == 0
                )



def add_vehicle_incident_constraint(model, y, scenario):
    """
    Ensure each vehicle serves at most one incident.
    """

    for vehicle in scenario["vehicles"]:

        vehicle_id = vehicle["vehicle_id"]

        model.Add(
            sum(
                y[(vehicle_id, incident["incident_id"])]
                for incident in scenario["incidents"]
            ) <= 1
        )


def dimensions_fit(device, vehicle):
    """
    Check whether an individual device fits inside a vehicle.
    Rotation is allowed by sorting dimensions.
    """

    device_dims = sorted(
        [
            device["length"],
            device["width"],
            device["height"]
        ],
        reverse=True
    )

    vehicle_dims = sorted(
        [
            vehicle["loading_length"],
            vehicle["loading_width"],
            vehicle["loading_height"]
        ],
        reverse=True
    )

    return all(
        device_dim <= vehicle_dim
        for device_dim, vehicle_dim in zip(device_dims, vehicle_dims)
    )


def add_individual_fit_constraint(model, x, scenario):
    """
    Prevent assigning a device to a vehicle if it does not fit.
    """

    for device in scenario["devices"]:
        for vehicle in scenario["vehicles"]:

            device_id = device["device_id"]
            vehicle_id = vehicle["vehicle_id"]

            if not dimensions_fit(device, vehicle):
                model.Add(x[(device_id, vehicle_id)] == 0)


def add_combined_capacity_constraint(model, x, scenario):
    """
    Ensure the combined dimensions of devices assigned
    to a vehicle do not exceed its loading capacity.
    """

    # Scaling supports decimal dimensions in CP-SAT.
    scale = 1000

    for vehicle in scenario["vehicles"]:

        vehicle_id = vehicle["vehicle_id"]

        vehicle_dims = sorted(
            [
                vehicle["loading_length"],
                vehicle["loading_width"],
                vehicle["loading_height"]
            ],
            reverse=True
        )

        # Check each of the three dimensions.
        for dimension_index in range(3):

            total_device_dimension = []

            for device in scenario["devices"]:

                device_id = device["device_id"]

                device_dims = sorted(
                    [
                        device["length"],
                        device["width"],
                        device["height"]
                    ],
                    reverse=True
                )

                total_device_dimension.append(
                    round(device_dims[dimension_index] * scale)
                    * x[(device_id, vehicle_id)]
                )

            model.Add(
                sum(total_device_dimension)
                <= round(vehicle_dims[dimension_index] * scale)
            )


def add_radio_constraint(model, x, y, scenario):
    """
    Every vehicle assigned to an incident must carry
    at least one Radio device.
    """

    radio_devices = [
        device
        for device in scenario["devices"]
        if device["device_type"] == "Radio"
    ]

    for vehicle in scenario["vehicles"]:

        vehicle_id = vehicle["vehicle_id"]

        assigned_radios = sum(
            x[(device["device_id"], vehicle_id)]
            for device in radio_devices
        )

        deployed = sum(
            y[(vehicle_id, incident["incident_id"])]
            for incident in scenario["incidents"]
        )

        model.Add(assigned_radios >= deployed)


def add_device_vehicle_link_constraint(model, x, y, scenario):
    """
    A device can only be assigned to a vehicle
    if that vehicle is assigned to an incident.
    """

    for device in scenario["devices"]:
        device_id = device["device_id"]

        for vehicle in scenario["vehicles"]:
            vehicle_id = vehicle["vehicle_id"]

            deployed = sum(
                y[(vehicle_id, incident["incident_id"])]
                for incident in scenario["incidents"]
            )

            model.Add(
                x[(device_id, vehicle_id)] <= deployed
            )


def create_device_incident_variables(model, x, y, scenario):
    """
    Create variables connecting each device assignment
    to the incident served by its vehicle.

    w[d, v, i] = 1 if device d is assigned to vehicle v
    and vehicle v serves incident i.
    """

    w = {}

    for device in scenario["devices"]:
        device_id = device["device_id"]

        for vehicle in scenario["vehicles"]:
            vehicle_id = vehicle["vehicle_id"]

            for incident in scenario["incidents"]:
                incident_id = incident["incident_id"]

                w[(device_id, vehicle_id, incident_id)] = (
                    model.NewBoolVar(
                        f"device_{device_id}_vehicle_{vehicle_id}_incident_{incident_id}"
                    )
                )

                # w cannot be 1 unless the device is assigned
                # to the vehicle.
                model.Add(
                    w[(device_id, vehicle_id, incident_id)]
                    <= x[(device_id, vehicle_id)]
                )

                # w cannot be 1 unless the vehicle serves
                # the incident.
                model.Add(
                    w[(device_id, vehicle_id, incident_id)]
                    <= y[(vehicle_id, incident_id)]
                )

                # If both assignments are 1, w must be 1.
                model.Add(
                    w[(device_id, vehicle_id, incident_id)]
                    >= x[(device_id, vehicle_id)]
                    + y[(vehicle_id, incident_id)] - 1
                )

    return w


def add_incident_demand_constraints(model, w, scenario):
    """
    Prevent an incident from receiving more devices
    of a particular type than it requires.

    Radio is excluded because it is mandatory vehicle equipment.
    """

    for incident in scenario["incidents"]:

        incident_id = incident["incident_id"]

        for device_type, required_quantity in incident[
            "required_devices"
        ].items():

            if device_type == "Radio":
                continue

            assigned_devices = []

            for device in scenario["devices"]:

                if device["device_type"] == device_type:

                    device_id = device["device_id"]

                    for vehicle in scenario["vehicles"]:

                        vehicle_id = vehicle["vehicle_id"]

                        assigned_devices.append(
                            w[(device_id, vehicle_id, incident_id)]
                        )

            model.Add(
                sum(assigned_devices) <= required_quantity
            )


def create_coverage_variables(model, w, scenario):
    """
    Create variables to track:
    1. Whether each incident receives required devices.
    2. How many devices of each required type are fulfilled.

    Radio is excluded from incident demand fulfilment.
    """

    served = {}
    fulfilled = {}

    for incident in scenario["incidents"]:

        incident_id = incident["incident_id"]

        total_required = sum(
            quantity
            for device_type, quantity
            in incident["required_devices"].items()
            if device_type != "Radio"
        )

        # Whether this incident receives at least
        # one required non-Radio device.
        served[incident_id] = model.NewBoolVar(
            f"incident_{incident_id}_served"
        )

        incident_fulfilled = []

        for device_type, required_quantity in incident[
            "required_devices"
        ].items():

            if device_type == "Radio":
                continue

            fulfilled[
                (incident_id, device_type)
            ] = model.NewIntVar(
                0,
                required_quantity,
                f"fulfilled_{incident_id}_{device_type}"
            )

            assigned_devices = []

            for device in scenario["devices"]:

                if device["device_type"] == device_type:

                    device_id = device["device_id"]

                    for vehicle in scenario["vehicles"]:

                        vehicle_id = vehicle["vehicle_id"]

                        assigned_devices.append(
                            w[(device_id, vehicle_id, incident_id)]
                        )

            model.Add(
                fulfilled[(incident_id, device_type)]
                == sum(assigned_devices)
            )

            incident_fulfilled.append(
                fulfilled[(incident_id, device_type)]
            )

        total_fulfilled = sum(incident_fulfilled)

        # An incident is served if it receives
        # at least one required non-Radio device.
        if total_required > 0:

            model.Add(
                total_fulfilled >= served[incident_id]
            )

            model.Add(
                total_fulfilled
                <= total_required * served[incident_id]
            )

        else:
            model.Add(served[incident_id] == 0)

    return served, fulfilled


def maximize_incident_coverage(model, served):
    """
    Stage 1:
    Maximize the number of incidents receiving
    at least one required non-Radio device.

    After optimization, fix the maximum coverage
    for the following optimization stages.
    """

    model.Maximize(sum(served.values()))

    solver = cp_model.CpSolver()

    # Allow up to 60 seconds for this optimization stage.
    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            "Incident coverage optimization was not proven optimal. "
            f"Solver status: {solver.StatusName(status)}"
        )

    maximum_coverage = sum(
        solver.Value(variable)
        for variable in served.values()
    )

    # Preserve the maximum coverage for subsequent stages.
    model.Add(
        sum(served.values()) == maximum_coverage
    )

    print("\nSTAGE 1: INCIDENT COVERAGE")
    print("Solver status:", solver.StatusName(status))
    print("Maximum incidents served:", maximum_coverage)

    return solver, maximum_coverage

def maximize_demand_fulfilment(model, fulfilled):
    """
    Stage 2:
    Maximize the total number of fulfilled non-Radio
    device requirements while preserving maximum
    incident coverage from Stage 1.
    """

    model.Maximize(sum(fulfilled.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            "Demand fulfilment optimization was not proven optimal. "
            f"Solver status: {solver.StatusName(status)}"
        )

    maximum_fulfilled = sum(
        solver.Value(variable)
        for variable in fulfilled.values()
    )

    # Preserve the maximum fulfilment for Stage 3.
    model.Add(
        sum(fulfilled.values()) == maximum_fulfilled
    )

    print("\nSTAGE 2: DEVICE DEMAND FULFILMENT")
    print("Solver status:", solver.StatusName(status))
    print("Maximum devices fulfilled:", maximum_fulfilled)

    return solver, maximum_fulfilled

def minimize_combined_objective(
    model,
    x,
    y,
    scenario,
    distance_weight=1.0,
    operating_cost_weight=1.0
):
    """
    Stage 3:
    Minimize travel distance and device operating costs
    while preserving the optimal results from Stages 1 and 2.
    """

    SCALE = 1000
    WEIGHT_SCALE = 1000

    objective_terms = []

    # Travel distance component
    for vehicle in scenario["vehicles"]:
        for incident in scenario["incidents"]:

            distance = math.sqrt(
                (vehicle["x"] - incident["x"]) ** 2
                + (vehicle["y"] - incident["y"]) ** 2
                + (vehicle["z"] - incident["z"]) ** 2
            )

            distance_coefficient = round(
                distance * SCALE
            )

            weight = round(
                distance_weight * WEIGHT_SCALE
            )

            coefficient = distance_coefficient * weight

            objective_terms.append(
                coefficient * y[
                    (vehicle["vehicle_id"], incident["incident_id"])
                ]
            )

    # Device operating cost component
    for device in scenario["devices"]:
        for vehicle in scenario["vehicles"]:

            cost_coefficient = round(
                device["operating_cost"] * SCALE
            )

            weight = round(
                operating_cost_weight * WEIGHT_SCALE
            )

            coefficient = cost_coefficient * weight

            objective_terms.append(
                coefficient * x[
                    (device["device_id"], vehicle["vehicle_id"])
                ]
            )

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            "Combined objective optimization was not proven optimal. "
            f"Solver status: {solver.StatusName(status)}"
        )

    print("\nSTAGE 3: COMBINED OBJECTIVE")
    print("Solver status:", solver.StatusName(status))
    print("Minimum scaled objective:", solver.ObjectiveValue())
    print("Distance weight:", distance_weight)
    print("Operating cost weight:", operating_cost_weight)

    return solver

def solve_reference_scenario(
    scenario,
    distance_weight=1.0,
    operating_cost_weight=1.0
):
    """
    Solve a planning scenario using the three-stage CP-SAT
    reference optimization process.

    Returns the optimized solution, validation results,
    objective metrics, and optimization statistics.
    """

    print("\n" + "=" * 55)
    print("CP-SAT REFERENCE SOLVER")
    print("Scenario:", scenario["scenario_id"])
    print("=" * 55)

    # 1. Build the model and decision variables
    model, x, y = build_reference_model(scenario)

    # 2. Add hard constraints
    add_device_uniqueness_constraint(model, x, scenario)
    add_storage_inventory_constraint(model, x, scenario)
    add_vehicle_incident_constraint(model, y, scenario)
    add_individual_fit_constraint(model, x, scenario)
    add_combined_capacity_constraint(model, x, scenario)
    add_radio_constraint(model, x, y, scenario)
    add_device_vehicle_link_constraint(model, x, y, scenario)

    # 3. Create device-to-incident linking variables
    w = create_device_incident_variables(
        model, x, y, scenario
    )

    # 4. Create coverage and fulfilment variables
    served, fulfilled = create_coverage_variables(
        model, w, scenario
    )

    # 5. Prevent over-delivery
    add_incident_demand_constraints(
        model, w, scenario
    )

    print("\nMODEL CREATED")
    print("Device-vehicle variables:", len(x))
    print("Vehicle-incident variables:", len(y))
    print("Device-vehicle-incident variables:", len(w))
    print("Incident coverage variables:", len(served))
    print("Demand fulfilment variables:", len(fulfilled))

    # 6. Stage 1: Maximize incident coverage
    solver, maximum_coverage = maximize_incident_coverage(
        model, served
    )

    # 7. Stage 2: Maximize device demand fulfilment
    solver, maximum_fulfilled = maximize_demand_fulfilment(
        model, fulfilled
    )

    # 8. Stage 3: Minimize combined objective
    solver = minimize_combined_objective(
        model,
        x,
        y,
        scenario,
        distance_weight=distance_weight,
        operating_cost_weight=operating_cost_weight
    )

    # 9. Extract device assignments
    device_assignments = []

    for device in scenario["devices"]:
        for vehicle in scenario["vehicles"]:

            variable = x[
                (device["device_id"], vehicle["vehicle_id"])
            ]

            if solver.Value(variable) == 1:
                device_assignments.append(
                    DeviceAssignment(
                        device_id=device["device_id"],
                        vehicle_id=vehicle["vehicle_id"]
                    )
                )

    # 10. Extract vehicle-to-incident assignments
    vehicle_assignments = []

    for vehicle in scenario["vehicles"]:
        for incident in scenario["incidents"]:

            variable = y[
                (vehicle["vehicle_id"], incident["incident_id"])
            ]

            if solver.Value(variable) == 1:
                vehicle_assignments.append(
                    VehicleAssignment(
                        vehicle_id=vehicle["vehicle_id"],
                        incident_id=incident["incident_id"]
                    )
                )

    # 11. Create the PlanningSolution
    reference_solution = PlanningSolution(
        solution_id=f"CP_SAT_{scenario['scenario_id']}",
        scenario_id=scenario["scenario_id"],
        device_assignments=device_assignments,
        vehicle_assignments=vehicle_assignments
    )

    # 12. Validate the solution
    validation_result = validate_solution(
        scenario,
        reference_solution
    )

    if not validation_result["valid"]:
        raise RuntimeError(
            "CP-SAT generated a solution that failed validation: "
            f"{validation_result['violations']}"
        )

    # 13. Calculate the actual objective
    objective_result = calculate_objective(
        scenario,
        reference_solution,
        distance_weight=distance_weight,
        operating_cost_weight=operating_cost_weight
    )

    print("\nREFERENCE SOLUTION SUMMARY")
    print("Assigned devices:", len(device_assignments))
    print("Assigned vehicles:", len(vehicle_assignments))
    print("Maximum incidents served:", maximum_coverage)
    print("Maximum devices fulfilled:", maximum_fulfilled)
    print("Hard constraints passed:", validation_result["hard_constraints_passed"])
    print("Total distance:", objective_result["total_distance"])
    print("Operating cost:", objective_result["total_hourly_operating_cost"])
    print("Combined objective:", objective_result["total_objective"])

    return {
        "solution": reference_solution,
        "validation": validation_result,
        "objective": objective_result,
        "maximum_coverage": maximum_coverage,
        "maximum_fulfilled": maximum_fulfilled
    }


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    scenario = load_scenario(
        "data/sample/Scenario_001.json"
    )

    result = solve_reference_scenario(scenario)

    reference_solution = result["solution"]

    print("\nDEVICE ASSIGNMENTS")

    for assignment in reference_solution.device_assignments:
        print(
            assignment.device_id,
            "->",
            assignment.vehicle_id
        )

    print("\nVEHICLE ASSIGNMENTS")

    for assignment in reference_solution.vehicle_assignments:
        print(
            assignment.vehicle_id,
            "->",
            assignment.incident_id
        )

    print("\nVALIDATION")
    print(result["validation"])

    print("\nOBJECTIVE")
    print(result["objective"])
