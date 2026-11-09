
from dataclasses import dataclass
from typing import List


@dataclass
class Vehicle:
    vehicle_id: str

    # Geographic coordinates
    x: float
    y: float
    z: float

    # Available loading space
    loading_length: float
    loading_width: float
    loading_height: float

    # Devices associated with the vehicle
    device_ids: List[str]


# Artificial example for testing
vehicle_301 = Vehicle(
    vehicle_id="Vehicle_301",
    x=20.0,
    y=30.0,
    z=5.0,
    loading_length=200.0,
    loading_width=200.0,
    loading_height=100.0,
    device_ids=["Radio_1"]
)


if __name__ == "__main__":
    print("Vehicle ID:", vehicle_301.vehicle_id)

    print(
        "Coordinates:",
        vehicle_301.x,
        vehicle_301.y,
        vehicle_301.z
    )

    print(
        "Loading Space:",
        vehicle_301.loading_length,
        vehicle_301.loading_width,
        vehicle_301.loading_height
    )

    print("Associated Devices:", vehicle_301.device_ids)
