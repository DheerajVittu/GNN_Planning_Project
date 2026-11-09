
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv, GINEConv

from src.graph.create_hetero_graph import create_hetero_graph


# --------------------------------------------------
# 1. EDGE-AWARE HETEROGENEOUS GNN
# --------------------------------------------------

class EdgeAwareHeterogeneousGNN(nn.Module):

    def __init__(
        self,
        metadata,
        input_dimensions,
        hidden_channels=16
    ):

        super().__init__()

        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels

        # Project every node type into a common feature dimension.
        self.node_projections = nn.ModuleDict({

            node_type: nn.Linear(
                input_dimensions[node_type],
                hidden_channels
            )

            for node_type in self.node_types
        })

        # Define a separate message-passing layer for each relation.
        relation_convolutions = {}

        for edge_type in self.edge_types:

            relation = edge_type[1]

            if relation in ["requires", "can_serve"]:

                # These relationships have numerical edge attributes.
                edge_network = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )

                relation_convolutions[edge_type] = GINEConv(
                    nn=edge_network,
                    edge_dim=1
                )

            else:

                # Relationships without numerical edge attributes.
                relation_convolutions[edge_type] = SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels
                )

        self.conv = HeteroConv(
            relation_convolutions,
            aggr="sum"
        )

    # --------------------------------------------------
    # 2. FORWARD PASS
    # --------------------------------------------------

    def forward(self, graph):

        # Step A: Project node features.
        x_dict = {

            node_type: F.relu(
                self.node_projections[node_type](
                    graph[node_type].x
                )
            )

            for node_type in self.node_types
        }

        # Step B: Collect edge attributes where available.
        edge_attr_dict = {

            edge_type: graph[edge_type].edge_attr

            for edge_type in self.edge_types

            if "edge_attr" in graph[edge_type]
        }

        # Step C: Perform heterogeneous message passing.
        updated_features = self.conv(
            x_dict,
            graph.edge_index_dict,
            edge_attr_dict=edge_attr_dict
        )

        # Step D: Combine original and updated representations.
        embeddings = {}

        for node_type in self.node_types:

            if node_type in updated_features:

                embeddings[node_type] = F.relu(
                    x_dict[node_type]
                    + updated_features[node_type]
                )

            else:

                embeddings[node_type] = x_dict[node_type]

        return embeddings


# --------------------------------------------------
# 3. TEST EDGE-AWARE GNN
# --------------------------------------------------

if __name__ == "__main__":

    scenario_file = "data/sample/Scenario_001.json"

    # Build the heterogeneous graph.
    graph = create_hetero_graph(scenario_file)

    input_dimensions = {
        "incident": 3,
        "device": 8,
        "storage": 4,
        "vehicle": 6
    }

    # Create the edge-aware model.
    model = EdgeAwareHeterogeneousGNN(
        metadata=graph.metadata(),
        input_dimensions=input_dimensions,
        hidden_channels=16
    )

    model.eval()

    # Run the forward pass without calculating gradients.
    with torch.no_grad():
        embeddings = model(graph)

    # Display results.
    print("\n" + "=" * 55)
    print("EDGE-AWARE HETEROGENEOUS GNN")
    print("=" * 55)

    print("\nNODE EMBEDDING SHAPES")
    print("-" * 30)

    for node_type, embedding in embeddings.items():

        print(
            node_type,
            "->",
            tuple(embedding.shape)
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nMODEL INFORMATION")
    print("-" * 30)

    print("Hidden channels:", model.hidden_channels)
    print("Total parameters:", total_parameters)

    print("\nFORWARD PASS")
    print("-" * 30)

    print("Edge-aware GNN forward pass completed successfully!")

    print("\n" + "=" * 55)
