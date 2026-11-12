
import json

from ortools.sat.python import cp_model


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


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    scenario = load_scenario(
        "data/sample/Scenario_001.json"
    )

    model, x, y = build_reference_model(scenario)

    # Constraint 4: Each device can be assigned once.
    add_device_uniqueness_constraint(model, x, scenario)

    # Constraint 5: Each vehicle serves at most one incident.
    add_vehicle_incident_constraint(model, y, scenario)

    # Constraint 1: Individual device fit
    add_individual_fit_constraint(model, x, scenario)

    # Constraint 2: Combined vehicle loading capacity
    add_combined_capacity_constraint(model, x, scenario)

    # Constraint 3: Every deployed vehicle must carry a Radio
    add_radio_constraint(model, x, y, scenario)

    # Link device assignments to deployed vehicles
    add_device_vehicle_link_constraint(model, x, y, scenario)

    # Connect device assignments to the incidents served
    w = create_device_incident_variables(model, x, y, scenario)

    # Prevent over-delivery of requested device types
    add_incident_demand_constraints(model, w, scenario)

    print("Device-incident linking variables created successfully!")
    print("Incident demand constraints added successfully!")

    print("Individual device fit constraint added successfully!")
    print("Combined vehicle capacity constraint added successfully!")
    print("Radio constraint added successfully!")
    print("Device-vehicle link constraint added successfully!")
    print("\nREFERENCE MODEL CREATED")
    print("Device uniqueness constraint added successfully!")
    print("Vehicle incident constraint added successfully!")

    print("\nDECISION VARIABLES")
    print("Device-vehicle variables:", len(x))
    print("Vehicle-incident variables:", len(y))
    print("Device-vehicle-incident variables:", len(w))

    total_variables = len(x) + len(y) + len(w)

    print("Total decision variables:", total_variables)
