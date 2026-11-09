
from src.data.input_schema import (
    Incident,
    Device,
    Storage,
    Vehicle,
    PlanningScenario
)


# ==========================================
# Create sample incident
# ==========================================

incident_1 = Incident(
    incident_id="Incident_1",
    x=10.0,
    y=20.0,
    z=5.0,
    required_devices={
        "Pump": 2,
        "Radio": 1
    }
)


# ==========================================
# Create individual devices
# ==========================================

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

radio_1 = Device(
    device_id="Radio_1",
    device_type="Radio",
    operating_cost=10.0,
    length=20.0,
    width=10.0,
    height=5.0
)


# ==========================================
# Create storage
# ==========================================

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


# ==========================================
# Create vehicle
# ==========================================

vehicle_301 = Vehicle(
    vehicle_id="Vehicle_301",
    x=20.0,
    y=30.0,
    z=5.0,
    loading_length=200.0,
    loading_width=200.0,
    loading_height=100.0
)


# ==========================================
# Combine into a planning scenario
# ==========================================

scenario = PlanningScenario(
    scenario_id="Scenario_001",
    incidents=[incident_1],
    devices=[pump_1, pump_2, radio_1],
    storages=[storage_1],
    vehicles=[vehicle_301]
)


# ==========================================
# Validate and display
# ==========================================

is_valid, messages = scenario.validate()

print("SCENARIO:", scenario.scenario_id)

print("\nENTITY COUNTS")
print("Incidents:", len(scenario.incidents))
print("Devices:", len(scenario.devices))
print("Storages:", len(scenario.storages))
print("Vehicles:", len(scenario.vehicles))

print("\nINCIDENT REQUIREMENTS")
print(incident_1.required_devices)

print("\nSTORAGE INVENTORY")
print(storage_1.device_ids)

print("\nVALIDATION RESULT")
print("Valid:", is_valid)

for message in messages:
    print(message)
