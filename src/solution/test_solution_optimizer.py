
import json
import torch

from src.graph.create_hetero_graph import create_hetero_graph
from src.solution.assignment_decoder import decode_assignments
from src.solution.solution_optimizer import (
    evaluate_solution,
    select_best_solution,
    generate_candidate_solutions
)
from src.solution.solution_schema import (
    PlanningSolution,
    DeviceAssignment,
    VehicleAssignment
)
from src.solution.solution_optimizer import select_best_solution

# 1. Load scenario
scenario_file = "data/sample/Scenario_001.json"

with open(scenario_file, "r") as file:
    scenario = json.load(file)


# 2. Build heterogeneous graph
graph = create_hetero_graph(scenario_file)


# 3. Create dummy prediction scores
# These are only for testing, not trained GNN predictions.

device_vehicle_count = graph[
    "device", "can_be_transported_by", "vehicle"
].edge_index.shape[1]

vehicle_incident_count = graph[
    "vehicle", "can_serve", "incident"
].edge_index.shape[1]

predictions = {
    "device_vehicle_logits": torch.zeros(device_vehicle_count),
    "vehicle_incident_logits": torch.zeros(vehicle_incident_count)
}


# 4. Generate a feasible solution
solution = decode_assignments(
    graph=graph,
    predictions=predictions,
    scenario=scenario,
    solution_id="Optimizer_Test_001"
)


# 5. Evaluate the solution
result = evaluate_solution(
    scenario=scenario,
    solution=solution,
    distance_weight=1.0,
    operating_cost_weight=1.0
)


# 6. Display evaluation
print("\n========== SOLUTION EVALUATION ==========")

print("Feasible:", result["feasible"])

if result["feasible"]:

    print("Incidents served:", result["incidents_served"])

    print("Fulfilled devices:", result["fulfilled_devices"])

    print(
        "Incident coverage:",
        result["incident_coverage_percentage"], "%"
    )

    print(
        "Demand fulfilment:",
        result["fulfillment_percentage"], "%"
    )

    print("Total distance:", round(result["total_distance"], 2))

    print(
        "Hourly operating cost:",
        result["hourly_operating_cost"]
    )

    print(
        "Combined objective:",
        round(result["total_objective"], 2)
    )

    print("Ranking score:", result["ranking_score"])

else:
    print("The solution violates one or more hard constraints.")

    for name, check in result["validation"]["checks"].items():
        if not check["passed"]:
            print("Failed:", name)

print("\nOptimizer evaluation test completed.")

# --------------------------------------------------
# STEP 7: COMPARE TWO FEASIBLE SOLUTIONS
# --------------------------------------------------

candidate_b = PlanningSolution(
    solution_id="Candidate_B",
    scenario_id=scenario["scenario_id"],
    vehicle_assignments=[
        VehicleAssignment("Vehicle_1", "Incident_1"),
        VehicleAssignment("Vehicle_2", "Incident_2"),
        VehicleAssignment("Vehicle_3", "Incident_3")
    ],
    device_assignments=[
        DeviceAssignment("Device_6", "Vehicle_1"),
        DeviceAssignment("Device_7", "Vehicle_2"),
        DeviceAssignment("Device_8", "Vehicle_3"),
        DeviceAssignment("Device_11", "Vehicle_1"),
        DeviceAssignment("Device_12", "Vehicle_2"),
        DeviceAssignment("Device_9", "Vehicle_3"),
        DeviceAssignment("Device_3", "Vehicle_3")
    ]
)

result_b = evaluate_solution(scenario, candidate_b)

print("\n========== CANDIDATE COMPARISON ==========")

print("Candidate A:", result["ranking_score"])
print("Candidate B:", result_b["ranking_score"])

if result["ranking_score"] > result_b["ranking_score"]:
    print("Winner: Candidate A")
elif result["ranking_score"] < result_b["ranking_score"]:
    print("Winner: Candidate B")
else:
    print("Both candidates have equal ranking.")


# --------------------------------------------------
# STEP 8: AUTOMATIC BEST-SOLUTION SELECTION
# --------------------------------------------------

selection = select_best_solution(
    scenario=scenario,
    candidate_solutions=[
        solution,
        candidate_b
    ]
)

print("\n========== AUTOMATIC SOLUTION SELECTION ==========")

for candidate in selection["evaluated_candidates"]:
    print(
        candidate["solution_id"],
        "| Feasible:",
        candidate["feasible"]
    )

best_solution = selection["best_solution"]

if best_solution is not None:
    print("\nSelected solution:", best_solution.solution_id)

    print(
        "Selected objective:",
        round(
            selection["best_evaluation"]["total_objective"],
            2
        )
    )
else:
    print("No feasible solution found.")


# --------------------------------------------------
# STEP 9: TEST REJECTION OF INVALID CANDIDATES
# --------------------------------------------------

invalid_candidate = PlanningSolution(
    solution_id="Invalid_Candidate",
    scenario_id=scenario["scenario_id"],
    vehicle_assignments=[
        VehicleAssignment("Vehicle_1", "Incident_1"),
        VehicleAssignment("Vehicle_1", "Incident_2"),
    ],
    device_assignments=[
        DeviceAssignment("Device_6", "Vehicle_1")
    ]
)

selection_with_invalid = select_best_solution(
    scenario=scenario,
    candidate_solutions=[
        solution,
        candidate_b,
        invalid_candidate
    ]
)

print("\n========== INVALID CANDIDATE TEST ==========")

for candidate in selection_with_invalid["evaluated_candidates"]:
    print(
        candidate["solution_id"],
        "| Feasible:",
        candidate["feasible"]
    )

print(
    "\nSelected solution:",
    selection_with_invalid["best_solution"].solution_id
)

# ============================================================
# TEST: GENERATED CANDIDATE SOLUTIONS
# ============================================================

generated_candidates = generate_candidate_solutions(
    scenario=scenario,
    base_solution=solution,
    graph=graph,
    max_candidates=100
)

print("\n========== GENERATED CANDIDATES ==========")
print("Number of feasible candidates:", len(generated_candidates))

for candidate in generated_candidates[:10]:
    evaluation = evaluate_solution(scenario, candidate)

    print(
        candidate.solution_id,
        "| Fulfilled devices:", evaluation["fulfilled_devices"],
        "| Objective:", round(evaluation["total_objective"], 2),
        "| Ranking:", evaluation["ranking_score"]
    )

# ============================================================
# TEST: SELECT BEST FROM ALL GENERATED CANDIDATES
# ============================================================

candidate_pool = [solution] + generated_candidates

selection_result = select_best_solution(
    scenario=scenario,
    candidate_solutions=candidate_pool
)

best_solution = selection_result["best_solution"]
best_evaluation = selection_result["best_evaluation"]
evaluated_candidates = selection_result["evaluated_candidates"]

print("\n========== BEST SOLUTION FROM ALL CANDIDATES ==========")
print("Total candidates evaluated:", len(evaluated_candidates))

if best_solution is not None:
    print("Best solution:", best_solution.solution_id)
    print("Incidents served:", best_evaluation["incidents_served"])
    print("Devices fulfilled:", best_evaluation["fulfilled_devices"])
    print(
        "Demand fulfilment:",
        round(best_evaluation["fulfillment_percentage"], 2), "%"
    )
    print("Objective:", round(best_evaluation["total_objective"], 2))
else:
    print("No feasible solution found.")