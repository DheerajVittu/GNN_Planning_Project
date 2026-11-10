
import json

from src.solution.objective_function import calculate_objective
from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)


# Load the sample scenario.
with open("data/sample/Scenario_001.json", "r") as file:
    scenario = json.load(file)


# Create a small example planning solution.
solution = PlanningSolution(
    solution_id="Test_Solution_001",
    scenario_id="Scenario_001",

    device_assignments=[
        DeviceAssignment("Device_1", "Vehicle_1"),
        DeviceAssignment("Device_6", "Vehicle_1")
    ],

    vehicle_assignments=[
        VehicleAssignment("Vehicle_1", "Incident_1")
    ]
)


# Calculate the objective.
results = calculate_objective(
    scenario=scenario,
    solution=solution,
    distance_weight=1.0,
    operating_cost_weight=1.0
)


# Display results.
print("\nOBJECTIVE FUNCTION TEST")
print("=" * 50)

for key, value in results.items():
    print(f"{key}: {value}")

print("\nObjective function executed successfully!")
