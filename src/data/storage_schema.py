
from dataclasses import dataclass
from typing import List


@dataclass
class Storage:
    storage_id: str

    # Geographic coordinates
    x: float
    y: float
    z: float

    # Devices available at this storage location
    device_ids: List[str]


# Artificial example for testing
storage_1 = Storage(
    storage_id="Storage_1",
    x=10.0,
    y=15.0,
    z=5.0,
    device_ids=[
        "Pump_1",
        "Pump_2",
        "Radio_1"
    ]
)


if __name__ == "__main__":
    print("Storage ID:", storage_1.storage_id)
    print(
        "Coordinates:",
        storage_1.x,
        storage_1.y,
        storage_1.z
    )
    print("Available Devices:")

    for device in storage_1.device_ids:
        print("-", device)
