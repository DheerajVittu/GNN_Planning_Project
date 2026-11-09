
import torch

from src.graph.build_graph import build_graph_from_json
from src.encoder.node_encoder import encode_nodes
from src.encoder.edge_encoder import encode_edges


# --------------------------------------------------
# 1. FEATURE NORMALIZER
# --------------------------------------------------

class FeatureNormalizer:
    """
    Standardizes numerical node features and edge attributes.

    Statistics are calculated from training scenarios only
    and reused for validation and test scenarios.
    """

    def __init__(self):

        self.node_statistics = {}
        self.edge_statistics = {}

        self.is_fitted = False

        # Numerical feature columns for each node type.
        # Device columns 0-3 are categorical one-hot indicators.
        self.numerical_columns = {
            "incident": [0, 1, 2],
            "device": [4, 5, 6, 7],
            "storage": [0, 1, 2, 3],
            "vehicle": [0, 1, 2, 3, 4, 5]
        }

    # --------------------------------------------------
    # 2. CALCULATE MEAN AND STANDARD DEVIATION
    # --------------------------------------------------

    @staticmethod
    def calculate_statistics(tensor):

        mean = tensor.mean(dim=0)

        std = tensor.std(
            dim=0,
            unbiased=False
        )

        # Prevent division by zero for constant features.
        std = torch.where(
            std < 1e-8,
            torch.ones_like(std),
            std
        )

        return {
            "mean": mean,
            "std": std
        }

    # --------------------------------------------------
    # 3. FIT USING TRAINING DATA
    # --------------------------------------------------

    def fit(
        self,
        training_node_data_list,
        training_edge_data_list
    ):
        """
        Fit normalization statistics using training scenarios.

        Each list contains the encoded data for one scenario.
        """

        if not training_node_data_list:
            raise ValueError("No training node data provided.")

        if len(training_node_data_list) != len(training_edge_data_list):
            raise ValueError(
                "Node and edge training scenario counts do not match."
            )

        # ----------------------------------------------
        # A. FIT NODE FEATURE STATISTICS
        # ----------------------------------------------

        for node_type, columns in self.numerical_columns.items():

            training_values = []

            for node_data in training_node_data_list:

                features = node_data[node_type]["x"]

                selected_features = features[:, columns]

                if selected_features.shape[0] > 0:
                    training_values.append(selected_features)

            if not training_values:
                raise ValueError(
                    f"No training features found for {node_type}."
                )

            combined_features = torch.cat(
                training_values,
                dim=0
            )

            self.node_statistics[node_type] = (
                self.calculate_statistics(combined_features)
            )

        # ----------------------------------------------
        # B. FIT EDGE ATTRIBUTE STATISTICS
        # ----------------------------------------------

        edge_types = [
            ("incident", "requires", "device"),
            ("vehicle", "can_serve", "incident")
        ]

        for edge_type in edge_types:

            training_values = []

            for edge_data in training_edge_data_list:

                information = edge_data[edge_type]

                if "edge_attr" in information:

                    attributes = information["edge_attr"]

                    if attributes.shape[0] > 0:
                        training_values.append(attributes)

            if not training_values:
                raise ValueError(
                    f"No training edge attributes found for {edge_type}."
                )

            combined_attributes = torch.cat(
                training_values,
                dim=0
            )

            self.edge_statistics[edge_type] = (
                self.calculate_statistics(combined_attributes)
            )

        self.is_fitted = True

        return self

    # --------------------------------------------------
    # 4. NORMALIZE NODE FEATURES
    # --------------------------------------------------

    def transform_nodes(self, encoded_nodes):

        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before transformation."
            )

        normalized_nodes = {}

        for node_type, information in encoded_nodes.items():

            # Preserve node IDs and copy the original features.
            normalized_information = {
                "x": information["x"].clone(),
                "node_ids": list(information["node_ids"])
            }

            columns = self.numerical_columns[node_type]

            statistics = self.node_statistics[node_type]

            mean = statistics["mean"]
            std = statistics["std"]

            # Standardization: (value - mean) / std
            normalized_information["x"][:, columns] = (
                normalized_information["x"][:, columns] - mean
            ) / std

            normalized_nodes[node_type] = normalized_information

        return normalized_nodes

    # --------------------------------------------------
    # 5. NORMALIZE EDGE ATTRIBUTES
    # --------------------------------------------------

    def transform_edges(self, encoded_edges):

        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before transformation."
            )

        normalized_edges = {}

        for edge_type, information in encoded_edges.items():

            normalized_information = {
                "edge_index": information["edge_index"].clone()
            }

            if "edge_attr" in information:

                attributes = information["edge_attr"].clone()

                statistics = self.edge_statistics[edge_type]

                mean = statistics["mean"]
                std = statistics["std"]

                normalized_information["edge_attr"] = (
                    attributes - mean
                ) / std

            normalized_edges[edge_type] = normalized_information

        return normalized_edges

    # --------------------------------------------------
    # 6. TRANSFORM COMPLETE SCENARIO ENCODING
    # --------------------------------------------------

    def transform(self, encoded_nodes, encoded_edges):

        return (
            self.transform_nodes(encoded_nodes),
            self.transform_edges(encoded_edges)
        )


# --------------------------------------------------
# 7. DISPLAY NORMALIZATION RESULTS
# --------------------------------------------------

def print_normalization_summary(
    original_nodes,
    original_edges,
    normalized_nodes,
    normalized_edges
):

    print("\n" + "=" * 60)
    print("FEATURE NORMALIZATION RESULTS")
    print("=" * 60)

    print("\nNODE FEATURES")
    print("-" * 40)

    for node_type in original_nodes:

        original = original_nodes[node_type]["x"]
        normalized = normalized_nodes[node_type]["x"]

        print(f"\n{node_type.upper()}")

        print("Original shape:", tuple(original.shape))
        print("Normalized shape:", tuple(normalized.shape))

        if node_type == "device":

            print(
                "One-hot features preserved:",
                torch.equal(
                    original[:, :4],
                    normalized[:, :4]
                )
            )

        numerical_columns = (
            FeatureNormalizer().numerical_columns[node_type]
        )

        values = normalized[:, numerical_columns]

        print(
            "Normalized numerical mean:",
            values.mean(dim=0)
        )

        print(
            "Normalized numerical standard deviation:",
            values.std(dim=0, unbiased=False)
        )

    print("\nEDGE ATTRIBUTES")
    print("-" * 40)

    for edge_type, information in normalized_edges.items():

        if "edge_attr" in information:

            print(f"\n{edge_type}")

            print(
                "Original shape:",
                tuple(original_edges[edge_type]["edge_attr"].shape)
            )

            print(
                "Normalized shape:",
                tuple(information["edge_attr"].shape)
            )

            print("Normalized values:")
            print(information["edge_attr"])

    print("\n" + "=" * 60)


# --------------------------------------------------
# 8. TEST NORMALIZATION
# --------------------------------------------------

if __name__ == "__main__":

    scenario_file = "data/sample/Scenario_001.json"

    # Build graph and generate encoded features.
    graph = build_graph_from_json(scenario_file)

    encoded_nodes = encode_nodes(graph)

    encoded_edges = encode_edges(
        graph,
        encoded_nodes
    )

    # Fit the normalizer using this scenario as a
    # temporary demonstration training sample.
    normalizer = FeatureNormalizer()

    normalizer.fit(
        training_node_data_list=[encoded_nodes],
        training_edge_data_list=[encoded_edges]
    )

    # Transform the encoded data.
    normalized_nodes, normalized_edges = normalizer.transform(
        encoded_nodes,
        encoded_edges
    )

    # Display results.
    print_normalization_summary(
        encoded_nodes,
        encoded_edges,
        normalized_nodes,
        normalized_edges
    )

    print("\nNormalization completed successfully!")
