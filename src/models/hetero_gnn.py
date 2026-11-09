
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv

from src.graph.create_hetero_graph import create_hetero_graph


# --------------------------------------------------
# 1. HETEROGENEOUS GNN MODEL
# --------------------------------------------------

class HeterogeneousGNN(nn.Module):

    def __init__(
        self,
        metadata,
        input_dimensions,
        hidden_channels=16
    ):

        super().__init__()

        self.node_types, self.edge_types = metadata

        self.hidden_channels = hidden_channels

        # Project each node type into the same hidden dimension.
        self.node_projections = nn.ModuleDict({

            node_type: nn.Linear(
                input_dimensions[node_type],
                hidden_channels
            )

            for node_type in self.node_types
        })

        # Define message passing for every relationship.
        self.conv = HeteroConv(
            {
                edge_type: SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels
                )

                for edge_type in self.edge_types
            },
            aggr="sum"
        )

    # --------------------------------------------------
    # 2. FORWARD PASS
    # --------------------------------------------------

    def forward(self, graph):

        # Step A: Project input features.
        x_dict = {

            node_type: F.relu(
                self.node_projections[node_type](
                    graph[node_type].x
                )
            )

            for node_type in self.node_types
        }

        # Step B: Perform heterogeneous message passing.
        updated_features = self.conv(
            x_dict,
            graph.edge_index_dict
        )

        # Step C: Combine original and updated features.
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
# 3. TEST GNN WITH DYNAMIC GRAPH
# --------------------------------------------------

if __name__ == "__main__":

    # Load a dynamically generated heterogeneous graph.
    scenario_file = "data/sample/Scenario_001.json"

    graph = create_hetero_graph(scenario_file)

    # Input dimensions from our node encoder.
    input_dimensions = {
        "incident": 3,
        "device": 8,
        "storage": 4,
        "vehicle": 6
    }

    # Create GNN model.
    model = HeterogeneousGNN(
        metadata=graph.metadata(),
        input_dimensions=input_dimensions,
        hidden_channels=16
    )

    # Set model to evaluation mode.
    model.eval()

    # Generate node embeddings.
    with torch.no_grad():
        embeddings = model(graph)

    # Display results.
    print("\n" + "=" * 55)
    print("HETEROGENEOUS GNN RESULTS")
    print("=" * 55)

    print("\nINPUT GRAPH")
    print("-" * 30)

    print("Node types:", graph.node_types)
    print("Edge types:", graph.edge_types)

    print("\nOUTPUT EMBEDDING SHAPES")
    print("-" * 30)

    for node_type, embedding in embeddings.items():

        print(
            node_type,
            "->",
            tuple(embedding.shape)
        )

    print("\nMODEL INFORMATION")
    print("-" * 30)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Hidden channels:", model.hidden_channels)
    print("Total model parameters:", total_parameters)

    print("\nFORWARD PASS")
    print("-" * 30)

    print("GNN forward pass completed successfully!")

    print("\n" + "=" * 55)
