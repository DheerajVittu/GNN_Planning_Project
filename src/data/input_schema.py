
from dataclasses import dataclass, field
from typing import Dict, List


# ==========================================
# 1. INCIDENT
# ==========================================

@dataclass
class Incident:
    incident_id: str

    # Incident location
    x: float
    y: float
    z: float

    # Required device types and quantities
    # Example: {"Pump": 2, "Radio": 1}
    required_devices: Dict[str, int]

    # Optional information; not confirmed in the actual solver
    incident_category: str = "unknown"
    severity: int = 0


# ==========================================
# 2. DEVICE
# ==========================================

@dataclass
class Device:
    device_id: str
    device_type: str

    # Operating cost
    operating_cost: float

    # Physical dimensions
    length: float
    width: float
    height: float


# ==========================================
# 3. STORAGE
# ==========================================

@dataclass
class Storage:
    storage_id: str

    # Storage location
    x: float
    y: float
    z: float

    # IDs of individual physical devices
    device_ids: List[str] = field(default_factory=list)


# ==========================================
# 4. VEHICLE
# ==========================================

@dataclass
class Vehicle:
    vehicle_id: str

    # Vehicle location
    x: float
    y: float
    z: float

    # Available loading dimensions
    loading_length: float
    loading_width: float
    loading_height: float


# ==========================================
# 5. COMPLETE PLANNING SCENARIO
# ==========================================

@dataclass
class PlanningScenario:
    scenario_id: str

    incidents: List[Incident]
    devices: List[Device]
    storages: List[Storage]
    vehicles: List[Vehicle]

    # ======================================
    # Validation
    # ======================================

    def validate(self):

        errors = []

        # Check unique IDs
        entity_groups = {
            "incident": self.incidents,
            "device": self.devices,
            "storage": self.storages,
            "vehicle": self.vehicles
        }

        for entity_type, entities in entity_groups.items():

            ids = [entity.__dict__[f"{entity_type}_id"] for entity in entities]

            if len(ids) != len(set(ids)):
                errors.append(
                    f"Duplicate {entity_type} IDs found."
                )

        # Check incident requirements
        for incident in self.incidents:

            for device_type, quantity in incident.required_devices.items():

                if quantity < 0:
                    errors.append(
                        f"Invalid required quantity for {device_type} "
                        f"in {incident.incident_id}"
                    )

        # Check device attributes
        for device in self.devices:

            if device.operating_cost < 0:
                errors.append(
                    f"Negative operating cost: {device.device_id}"
                )

            dimensions = [
                device.length,
                device.width,
                device.height
            ]

            if any(value <= 0 for value in dimensions):
                errors.append(
                    f"Invalid dimensions: {device.device_id}"
                )

        # Check storage inventory references
        device_ids = {
            device.device_id for device in self.devices
        }

        for storage in self.storages:

            for device_id in storage.device_ids:

                if device_id not in device_ids:
                    errors.append(
                        f"{device_id} in {storage.storage_id} "
                        "does not exist in the device list."
                    )

        # Check each device belongs to at most one storage
        assigned_devices = []

        for storage in self.storages:
            assigned_devices.extend(storage.device_ids)

        if len(assigned_devices) != len(set(assigned_devices)):
            errors.append(
                "A device is listed in multiple storage locations."
            )

        # Check vehicle loading dimensions
        for vehicle in self.vehicles:

            dimensions = [
                vehicle.loading_length,
                vehicle.loading_width,
                vehicle.loading_height
            ]

            if any(value <= 0 for value in dimensions):
                errors.append(
                    f"Invalid loading dimensions: {vehicle.vehicle_id}"
                )

        # Return validation result
        if errors:
            return False, errors

        return True, ["Scenario validation successful!"]
