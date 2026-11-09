
from dataclasses import dataclass
from typing import Dict


@dataclass
class Incident:
    incident_id: str

    # Geographic coordinates
    x: float
    y: float
    z: float

    # Required device types and quantities
    required_devices: Dict[str, int]


# Artificial example for testing our schema
incident_1 = Incident(
    incident_id="Incident_1",
    x=10.0,
    y=20.0,
    z=5.0,
    required_devices={
        "Pump": 2,
        "Emergency Raft": 1,
        "Radio": 1
    }
)

if __name__ == "__main__":
    print("Incident:", incident_1.incident_id)
    print("Coordinates:", incident_1.x, incident_1.y, incident_1.z)
    print("Required devices:", incident_1.required_devices)
