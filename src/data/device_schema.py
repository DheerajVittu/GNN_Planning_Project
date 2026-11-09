
from dataclasses import dataclass


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


# Artificial example for testing
pump_1 = Device(
    device_id="Pump_1",
    device_type="Pump",
    operating_cost=30.0,
    length=150.0,
    width=50.0,
    height=60.0
)

pump_2 = Device(
    device_id="Pump_2",
    device_type="Pump",
    operating_cost=30.0,
    length=150.0,
    width=50.0,
    height=60.0
)


if __name__ == "__main__":
    print("Device ID:", pump_1.device_id)
    print("Device Type:", pump_1.device_type)
    print("Operating Cost:", pump_1.operating_cost)
    print(
        "Dimensions:",
        pump_1.length,
        pump_1.width,
        pump_1.height
    )
