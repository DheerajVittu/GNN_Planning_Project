
import json
import torch

from src.graph.create_hetero_graph import create_hetero_graph
from src.solution.assignment_decoder import decode_assignments
from src.solution.constraint_validator import validate_solution
from src.solution.objective_function import calculate_objective

# --------------------------------------------------
# STEP 1: LOAD THE SAMPLE SCENARIO
# --------------------------------------------------

scenario_file = "data/sample/Scenario_001.json"

with open(scenario_file, "r") as file:
    scenario = json.load(file)


# --------------------------------------------------
# STEP 2: CREATE THE ORIGINAL HETEROGENEOUS GRAPH
# --------------------------------------------------

graph = create_hetero_graph(scenario_file)


# --------------------------------------------------
# STEP 3: CREATE DUMMY PREDICTION SCORES
# --------------------------------------------------
# These are not real GNN predictions.
# We use zero logits to test decoder functionality.

device_vehicle_edge_count = graph[
    "device",
    "can_be_transported_by",
    "vehicle"
].edge_index.shape[1]

vehicle_incident_edge_count = graph[
    "vehicle",
    "can_serve",
    "incident"
].edge_index.shape[1]

predictions = {
    "device_vehicle_logits": torch.zeros(
        device_vehicle_edge_count
    ),
    "vehicle_incident_logits": torch.zeros(
        vehicle_incident_edge_count
    )
}


# --------------------------------------------------
# STEP 4: RUN THE CONSTRAINT-AWARE DECODER
# --------------------------------------------------

solution = decode_assignments(
    graph=graph,
    predictions=predictions,
    scenario=scenario,
    solution_id="Decoder_Test_001"
)


# --------------------------------------------------
# STEP 5: DISPLAY THE ASSIGNMENTS
# --------------------------------------------------

print("\n========== DECODER RESULTS ==========")

print("\nVehicle Assignments:")

for assignment in solution.vehicle_assignments:
    print(
        assignment.vehicle_id,
        "->",
        assignment.incident_id
    )

print("\nDevice Assignments:")

for assignment in solution.device_assignments:
    print(
        assignment.device_id,
        "->",
        assignment.vehicle_id
    )


# --------------------------------------------------
# STEP 6: VALIDATE THE GENERATED SOLUTION
# --------------------------------------------------

result = validate_solution(scenario, solution)

objective = calculate_objective(
    scenario=scenario,
    solution=solution,
    distance_weight=1.0,
    operating_cost_weight=1.0
)

print("\n========== CONSTRAINT VALIDATION ==========")

for name, check in result["checks"].items():

    status = "PASS" if check["passed"] else "FAIL"

    print(f"{name}: {status}")

    for message in check["violations"]:
        print("  -", message)


# --------------------------------------------------
# STEP 7: DISPLAY FINAL RESULT
# --------------------------------------------------

print("\n========== FINAL RESULT ==========")

print(
    "Hard constraints passed:",
    result["hard_constraints_passed"]
)

print(
    "Overall valid:",
    result["valid"]
)

print(
    "Total assigned devices:",
    len(solution.device_assignments)
)

print(
    "Total deployed vehicles:",
    len(solution.vehicle_assignments)
)


print("\nDemand Summary:")

for incident_id, information in result["demand_summary"].items():
    print(f"\n{incident_id}")
    print("Required:", information["required"])
    print("Assigned:", information["assigned"])
    print("Fulfilled:", information["fulfilled"])
    print(
        "Fulfilment:",
        f'{information["fulfillment_percentage"]}%'
    )
    print(
        "Unmet demand:",
        information["unmet_excluding_radio"]
    )

# Display overall coverage statistics.
coverage = result["overall_coverage"]

print("\n========== OVERALL COVERAGE ==========")

print(
    "Incidents receiving devices:",
    f'{coverage["incidents_receiving_devices"]}/'
    f'{coverage["total_incidents"]}'
)

print(
    "Incident coverage:",
    f'{coverage["incident_coverage_percentage"]}%'
)

print(
    "Total required devices:",
    coverage["total_required_excluding_radio"]
)

print(
    "Total fulfilled devices:",
    coverage["total_fulfilled_excluding_radio"]
)

print(
    "Total unmet devices:",
    coverage["total_unmet_excluding_radio"]
)

print(
    "Overall fulfilment:",
    f'{coverage["overall_fulfillment_percentage"]}%'
)

print("\nDecoder test completed.")

print("\n========== OBJECTIVE EVALUATION ==========")

print(
    "Total vehicle distance:",
    round(objective["total_distance"], 2)
)

print(
    "Total hourly operating cost:",
    round(objective["total_hourly_operating_cost"], 2)
)

print(
    "Weighted distance:",
    round(objective["weighted_distance"], 2)
)

print(
    "Weighted operating cost:",
    round(objective["weighted_operating_cost"], 2)
)

print(
    "Combined objective:",
    round(objective["total_objective"], 2)
)