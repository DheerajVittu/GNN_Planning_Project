
import json

from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)
from src.solution.constraint_validator import validate_solution


with open("data/sample/Scenario_001.json", "r") as file:
    scenario = json.load(file)


# Deliberately invalid:
# Device_1 is available in storage.
# Device_6 is available in storage.
# Device_14 is available in storage.
#
# To test an unavailable device, remove Device_14
# from every storage inventory before validation.

for storage in scenario["storages"]:
    if "Device_14" in storage["device_ids"]:
        storage["device_ids"].remove("Device_14")


solution = PlanningSolution(
    solution_id="Invalid_Test_005",
    scenario_id=scenario["scenario_id"],
    device_assignments=[
        DeviceAssignment("Device_6", "Vehicle_1"),
        DeviceAssignment("Device_14", "Vehicle_1")
    ],
    vehicle_assignments=[
        VehicleAssignment("Vehicle_1", "Incident_1")
    ]
)


result = validate_solution(scenario, solution)

print("\n========== STORAGE AVAILABILITY TEST ==========")

for name, check in result["checks"].items():
    status = "PASS" if check["passed"] else "FAIL"
    print(f"{name}: {status}")

print("\nViolations:")
for violation in result["violations"]:
    print("-", violation)

print(
    "\nStorage violation correctly detected:",
    not result["checks"]["storage_stock_availability"]["passed"]
)
