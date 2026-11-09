
from dataclasses import dataclass, field
from typing import List


@dataclass
class DeviceAssignment:
    """
    Represents the assignment of one device to one vehicle.
    """
    device_id: str
    vehicle_id: str


@dataclass
class VehicleAssignment:
    """
    Represents the assignment of one vehicle to one incident.
    """
    vehicle_id: str
    incident_id: str


@dataclass
class PlanningSolution:
    """
    Stores the complete transportation plan generated for a scenario.

    A solution contains:
    1. Device-to-vehicle assignments.
    2. Vehicle-to-incident assignments.
    """

    solution_id: str
    scenario_id: str

    device_assignments: List[DeviceAssignment] = field(
        default_factory=list
    )

    vehicle_assignments: List[VehicleAssignment] = field(
        default_factory=list
    )

    def validate_references(self, scenario):
        """
        Checks whether the assignments refer to entities
        that actually exist in the input scenario.

        This is an initial structural check.
        The six hard constraints will be checked separately.
        """

        messages = []

        if self.scenario_id != scenario.scenario_id:
            messages.append(
                "Solution scenario ID does not match input scenario."
            )

        valid_device_ids = {
            device.device_id for device in scenario.devices
        }

        valid_vehicle_ids = {
            vehicle.vehicle_id for vehicle in scenario.vehicles
        }

        valid_incident_ids = {
            incident.incident_id for incident in scenario.incidents
        }

        # Check device assignments
        assigned_devices = set()

        for assignment in self.device_assignments:

            if assignment.device_id not in valid_device_ids:
                messages.append(
                    f"Unknown device: {assignment.device_id}"
                )

            if assignment.vehicle_id not in valid_vehicle_ids:
                messages.append(
                    f"Unknown vehicle: {assignment.vehicle_id}"
                )

            if assignment.device_id in assigned_devices:
                messages.append(
                    f"Device assigned more than once: "
                    f"{assignment.device_id}"
                )

            assigned_devices.add(assignment.device_id)

        # Check vehicle assignments
        assigned_vehicles = set()

        for assignment in self.vehicle_assignments:

            if assignment.vehicle_id not in valid_vehicle_ids:
                messages.append(
                    f"Unknown vehicle: {assignment.vehicle_id}"
                )

            if assignment.incident_id not in valid_incident_ids:
                messages.append(
                    f"Unknown incident: {assignment.incident_id}"
                )

            if assignment.vehicle_id in assigned_vehicles:
                messages.append(
                    f"Vehicle assigned to multiple incidents: "
                    f"{assignment.vehicle_id}"
                )

            assigned_vehicles.add(assignment.vehicle_id)

        return len(messages) == 0, messages
