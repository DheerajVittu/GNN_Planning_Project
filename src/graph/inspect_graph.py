from build_graph import G

print("GRAPH SUMMARY")
print("====================")

print("\nNode types:")
for node, attributes in G.nodes(data=True):
    print(node, ":", attributes["node_type"])

print("\nRelationships:")
for source, target, attributes in G.edges(data=True):
    print(
        attributes["relation"],
        ":",
        source,
        "->",
        target
    )

print("\nNodes grouped by type:")

node_types = {}

for node, attributes in G.nodes(data=True):
    node_type = attributes["node_type"]

    if node_type not in node_types:
        node_types[node_type] = []

    node_types[node_type].append(node)

for node_type, nodes in node_types.items():
    print(node_type, ":", nodes)