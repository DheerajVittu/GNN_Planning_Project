
import json

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)
from src.solution.constraint_validator import validate_solution


# Load the sample scenario.
with open("data/sample/Scenario_001.json", "r") as file:
    scenario = json.load(file)


# Create a small test solution.
solution = PlanningSolution(
    solution_id="Test_Solution_001",
    scenario_id=scenario["scenario_id"],
    device_assignments=[
        DeviceAssignment(
            device_id="Device_1",
            vehicle_id="Vehicle_1"
        ),
        DeviceAssignment(
            device_id="Device_6",
            vehicle_id="Vehicle_1"
        )
    ],
    vehicle_assignments=[
        VehicleAssignment(
            vehicle_id="Vehicle_1",
            incident_id="Incident_1"
        )
    ]
)


# Validate the solution.
result = validate_solution(scenario, solution)


print("\n========== CONSTRAINT VALIDATION ==========")

for constraint_name, check in result["checks"].items():
    status = "PASS" if check["passed"] else "FAIL"

    print(f"\n{constraint_name}: {status}")

    for message in check["violations"]:
        print("  -", message)

print("\n========== FINAL RESULT ==========")
print("Hard constraints passed:", result["hard_constraints_passed"])
print("Overall valid:", result["valid"])

print("\nDemand summary:")
print(json.dumps(result["demand_summary"], indent=4))

print("\nConstraint validation test completed.")
