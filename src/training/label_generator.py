
import torch


def create_assignment_labels(graph, reference_solution):
    """
    Create binary labels for the two GNN assignment prediction heads.

    Labels follow the exact order of candidate edges in the graph.
    1 = assignment exists in the CP-SAT reference solution.
    0 = candidate assignment does not exist in the reference solution.
    """

    # Build ID-to-index mappings using the graph's preserved node order.
    device_ids = graph["device"].node_ids
    vehicle_ids = graph["vehicle"].node_ids
    incident_ids = graph["incident"].node_ids

    device_to_index = {
        node_id: index for index, node_id in enumerate(device_ids)
    }
    vehicle_to_index = {
        node_id: index for index, node_id in enumerate(vehicle_ids)
    }
    incident_to_index = {
        node_id: index for index, node_id in enumerate(incident_ids)
    }

    # Read positive assignments from the CP-SAT reference solution.
    positive_device_vehicle = {
        (item["device_id"], item["vehicle_id"])
        for item in reference_solution["device_assignments"]
    }

    positive_vehicle_incident = {
        (item["vehicle_id"], item["incident_id"])
        for item in reference_solution["vehicle_assignments"]
    }

    # Get candidate device-to-vehicle edges.
    device_vehicle_edges = graph[
        "device", "can_be_transported_by", "vehicle"
    ].edge_index

    device_vehicle_labels = []

    for device_index, vehicle_index in device_vehicle_edges.t().tolist():
        pair = (
            device_ids[device_index],
            vehicle_ids[vehicle_index]
        )

        device_vehicle_labels.append(
            1.0 if pair in positive_device_vehicle else 0.0
        )

    # Get candidate vehicle-to-incident edges.
    vehicle_incident_edges = graph[
        "vehicle", "can_serve", "incident"
    ].edge_index

    vehicle_incident_labels = []

    for vehicle_index, incident_index in vehicle_incident_edges.t().tolist():
        pair = (
            vehicle_ids[vehicle_index],
            incident_ids[incident_index]
        )

        vehicle_incident_labels.append(
            1.0 if pair in positive_vehicle_incident else 0.0
        )

    # Convert labels into PyTorch tensors.
    labels = {
        "device_vehicle_labels": torch.tensor(
            device_vehicle_labels,
            dtype=torch.float
        ),
        "vehicle_incident_labels": torch.tensor(
            vehicle_incident_labels,
            dtype=torch.float
        )
    }

    # Verify that every reference assignment exists among candidate edges.
    if sum(device_vehicle_labels) != len(positive_device_vehicle):
        raise ValueError(
            "Some reference device-to-vehicle assignments "
            "are missing from the graph."
        )

    if sum(vehicle_incident_labels) != len(positive_vehicle_incident):
        raise ValueError(
            "Some reference vehicle-to-incident assignments "
            "are missing from the graph."
        )

    return labels
