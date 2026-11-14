
import json
from dataclasses import asdict
from pathlib import Path

from src.data.scenario_generator import (
    generate_scenario,
    save_scenario
)

from src.optimization.reference_solver import (
    load_scenario,
    solve_reference_scenario
)


def generate_reference_example(
    num_devices,
    num_vehicles,
    num_storages,
    num_incidents,
    seed
):
    """Generate a scenario and solve it using CP-SAT."""

    scenario_id = (
        f"Scenario_D{num_devices}"
        f"_V{num_vehicles}"
        f"_I{num_incidents}"
        f"_S{seed}"
    )

    print("\nGENERATING SCENARIO:", scenario_id, flush=True)

    # Generate scenario
    generated_scenario = generate_scenario(
        scenario_id=scenario_id,
        num_incidents=num_incidents,
        num_storages=num_storages,
        num_vehicles=num_vehicles,
        seed=seed,
        num_devices=num_devices
    )

    # Validate scenario
    is_valid, messages = generated_scenario.validate()

    if not is_valid:
        raise ValueError(f"Invalid scenario: {messages}")

    print("Scenario validation: PASSED", flush=True)

    # Save scenario
    scenario_path = save_scenario(
        generated_scenario,
        output_directory="data/generated/scenarios"
    )

    print("Scenario saved:", scenario_path, flush=True)

    # Load scenario
    scenario = load_scenario(scenario_path)

    # Solve using CP-SAT
    result = solve_reference_scenario(scenario)

    # Prepare reference data
    reference_data = {
        "scenario_id": scenario_id,
        "scenario_file": str(scenario_path),
        "reference_solution": asdict(result["solution"]),
        "validation": result["validation"],
        "objective": result["objective"],
        "maximum_coverage": result["maximum_coverage"],
        "maximum_fulfilled": result["maximum_fulfilled"]
    }

    # Save reference solution
    output_directory = Path("data/generated/references")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory / f"{scenario_id}_reference.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(reference_data, file, indent=4)

    print("\nREFERENCE DATA SAVED:", output_path, flush=True)

    return result


if __name__ == "__main__":

    generate_reference_example(
        num_devices=200,
        num_vehicles=50,
        num_storages=10,
        num_incidents=2,
        seed=42
    )
