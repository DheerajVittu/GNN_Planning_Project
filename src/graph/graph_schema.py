
# src/graph/graph_schema.py

# Node types in our planning problem
NODE_TYPES = [
    "incident",
    "device",
    "storage",
    "vehicle"
]

# Proposed relationships between node types
EDGE_TYPES = [
    ("storage", "stores", "device"),
    ("incident", "requires", "device"),
    ("device", "can_be_transported_by", "vehicle"),
    ("vehicle", "can_serve", "incident")
]


if __name__ == "__main__":
    print("Node Types:")

    for node in NODE_TYPES:
        print("-", node)

    print("\nEdge Types:")

    for source, relation, target in EDGE_TYPES:
        print(f"{source} --{relation}--> {target}")
