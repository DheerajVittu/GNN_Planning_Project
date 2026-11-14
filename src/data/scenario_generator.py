
import json
import random

from dataclasses import asdict
from pathlib import Path

from src.data.input_schema import (
    Incident,
    Device,
    Storage,
    Vehicle,
    PlanningScenario
)


# --------------------------------------------------
# 1. ILLUSTRATIVE DEVICE SPECIFICATIONS
# --------------------------------------------------
# These are temporary sample values, not real equipment data.
# We will replace them when actual project data is available.

DEVICE_CATALOG = {
    "Pump": {
        "operating_cost": 30.0,
        "length": 150.0,
        "width": 50.0,
        "height": 60.0
    },
    "Radio": {
        "operating_cost": 10.0,
        "length": 20.0,
        "width": 10.0,
        "height": 5.0
    },
    "MedicalKit": {
        "operating_cost": 15.0,
        "length": 40.0,
        "width": 30.0,
        "height": 20.0
    },
    "EmergencyRaft": {
        "operating_cost": 45.0,
        "length": 180.0,
        "width": 80.0,
        "height": 50.0
    }
}


# --------------------------------------------------
# 2. SCENARIO GENERATOR
# --------------------------------------------------

def generate_scenario(
    scenario_id="Scenario_001",
    num_incidents=3,
    num_storages=2,
    num_vehicles=3,
    seed=42,
    num_devices=None
):
    """
    Generate a synthetic disaster-response planning scenario.

    The generated scenario contains:
    - Incidents and their device requirements
    - Individual devices
    - Storage locations and their inventories
    - Vehicles and their loading capacities

    A fixed random seed makes the scenario reproducible.
    """

    rng = random.Random(seed)

    # --------------------------------------------------
    # STEP A: Generate incident requirements
    # --------------------------------------------------

    incidents = []
    total_required_devices = {
        device_type: 0 for device_type in DEVICE_CATALOG
    }

    for i in range(1, num_incidents + 1):

        # Each incident requires at least one device type.
        selected_types = rng.sample(
            list(DEVICE_CATALOG.keys()),
            k=rng.randint(1, 3)
        )

        required_devices = {}

        for device_type in selected_types:
            quantity = rng.randint(1, 2)

            required_devices[device_type] = quantity

            total_required_devices[device_type] += quantity

        incident = Incident(
            incident_id=f"Incident_{i}",
            x=round(rng.uniform(0, 100), 2),
            y=round(rng.uniform(0, 100), 2),
            z=round(rng.uniform(0, 20), 2),
            required_devices=required_devices,
            incident_category=rng.choice(
                ["Flood", "Fire", "Medical", "Rescue"]
            ),
            severity=rng.randint(1, 5)
        )

        incidents.append(incident)

        # --------------------------------------------------
    # STEP B: Generate individual devices
    # --------------------------------------------------

    devices = []

    device_counter = 1

    # Preserve the original generation behaviour
    # when no exact device count is provided.
    if num_devices is None:

        device_quantities = {
            device_type: required_quantity + rng.randint(1, 2)
            for device_type, required_quantity
            in total_required_devices.items()
        }

    else:

        if num_devices < 0:
            raise ValueError("num_devices cannot be negative.")

        minimum_required = sum(total_required_devices.values())

        if num_devices < minimum_required:
            raise ValueError(
                f"Requested {num_devices} devices, but "
                f"incident requirements total {minimum_required}."
            )

        # Start with enough devices to meet all requirements.
                # Start with enough devices to meet all incident requirements.
        device_quantities = dict(total_required_devices)

        # Every potentially deployed vehicle needs a Radio.
        additional_radios = max(
            0,
            num_vehicles - device_quantities["Radio"]
        )

        remaining_devices = num_devices - minimum_required

        if additional_radios > remaining_devices:
            raise ValueError(
                "Not enough devices to provide Radios for all vehicles "
                "while meeting incident requirements."
            )

        device_quantities["Radio"] += additional_radios
        remaining_devices -= additional_radios

        # Distribute the remaining devices randomly.
        for _ in range(remaining_devices):
            selected_type = rng.choice(list(DEVICE_CATALOG.keys()))
            device_quantities[selected_type] += 1

    # Create the individual device objects.
    for device_type, quantity in device_quantities.items():

        specifications = DEVICE_CATALOG[device_type]

        for _ in range(quantity):

            device = Device(
                device_id=f"Device_{device_counter}",
                device_type=device_type,
                operating_cost=specifications["operating_cost"],
                length=specifications["length"],
                width=specifications["width"],
                height=specifications["height"]
            )

            devices.append(device)
            device_counter += 1

    # --------------------------------------------------
    # STEP C: Generate storage locations
    # --------------------------------------------------

    storages = []

    storage_device_ids = [
        [] for _ in range(num_storages)
    ]

    # Distribute every device across the available storages.
    for index, device in enumerate(devices):
        storage_index = index % num_storages
        storage_device_ids[storage_index].append(device.device_id)

    for i in range(num_storages):

        storage = Storage(
            storage_id=f"Storage_{i + 1}",
            x=round(rng.uniform(0, 100), 2),
            y=round(rng.uniform(0, 100), 2),
            z=round(rng.uniform(0, 20), 2),
            device_ids=storage_device_ids[i]
        )

        storages.append(storage)

    # --------------------------------------------------
    # STEP D: Generate vehicles
    # --------------------------------------------------

    vehicles = []

    for i in range(1, num_vehicles + 1):

        vehicle = Vehicle(
            vehicle_id=f"Vehicle_{i}",
            x=round(rng.uniform(0, 100), 2),
            y=round(rng.uniform(0, 100), 2),
            z=round(rng.uniform(0, 20), 2),
            loading_length=rng.choice([200.0, 250.0, 300.0]),
            loading_width=rng.choice([100.0, 150.0, 200.0]),
            loading_height=rng.choice([100.0, 150.0, 200.0])
        )

        vehicles.append(vehicle)

    # --------------------------------------------------
    # STEP E: Create the complete planning scenario
    # --------------------------------------------------

    scenario = PlanningScenario(
        scenario_id=scenario_id,
        incidents=incidents,
        devices=devices,
        storages=storages,
        vehicles=vehicles
    )

    return scenario


# --------------------------------------------------
# 3. SAVE SCENARIO AS JSON
# --------------------------------------------------

def save_scenario(scenario, output_directory="data/sample"):

    output_path = Path(output_directory)

    # Create the directory if it does not exist.
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{scenario.scenario_id}.json"

    # Convert dataclass objects into a JSON-compatible dictionary.
    scenario_data = asdict(scenario)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(scenario_data, file, indent=4)

    return file_path


# --------------------------------------------------
# 4. PRINT SCENARIO SUMMARY
# --------------------------------------------------

def print_scenario_summary(scenario, file_path):

    print("\n" + "=" * 55)
    print("SYNTHETIC PLANNING SCENARIO GENERATED")
    print("=" * 55)

    print(f"\nScenario ID: {scenario.scenario_id}")

    print("\nENTITY COUNTS")
    print("-" * 30)
    print(f"Incidents : {len(scenario.incidents)}")
    print(f"Devices   : {len(scenario.devices)}")
    print(f"Storages  : {len(scenario.storages)}")
    print(f"Vehicles  : {len(scenario.vehicles)}")

    print("\nINCIDENT REQUIREMENTS")
    print("-" * 30)

    for incident in scenario.incidents:
        print(
            f"{incident.incident_id}: "
            f"{incident.required_devices}"
        )

    print("\nDEVICE INVENTORY")
    print("-" * 30)

    for device in scenario.devices:
        print(
            f"{device.device_id}: "
            f"{device.device_type}"
        )

    print("\nSTORAGE INVENTORY")
    print("-" * 30)

    for storage in scenario.storages:
        print(
            f"{storage.storage_id}: "
            f"{storage.device_ids}"
        )

    print("\nVEHICLES")
    print("-" * 30)

    for vehicle in scenario.vehicles:
        print(
            f"{vehicle.vehicle_id}: "
            f"Loading capacity = "
            f"{vehicle.loading_length} x "
            f"{vehicle.loading_width} x "
            f"{vehicle.loading_height}"
        )

    print("\nVALIDATION")
    print("-" * 30)

    is_valid, messages = scenario.validate()

    print(f"Valid: {is_valid}")

    for message in messages:
        print(message)

    print("\nJSON FILE")
    print("-" * 30)
    print(file_path)

    print("\n" + "=" * 55)


# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    # Generate one reproducible sample scenario.
    generated_scenario = generate_scenario(
        scenario_id="Scenario_001",
        num_incidents=3,
        num_storages=2,
        num_vehicles=3,
        seed=42
    )

    # Validate the scenario before saving.
    is_valid, validation_messages = generated_scenario.validate()

    if not is_valid:
        print("Scenario validation failed:")

        for message in validation_messages:
            print(message)

    else:
        # Save the generated scenario.
        saved_file = save_scenario(generated_scenario)

        # Print its details.
        print_scenario_summary(
            generated_scenario,
            saved_file
        )
