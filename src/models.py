import torch
import torch.nn as nn


class LinearAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder for sensor anomaly detection.

    Input shape:
        batch_size, window_size, num_features

    Example:
        batch_size, 30, 21

    The model flattens the time window, compresses it into a bottleneck,
    then reconstructs it back to the original shape.
    """

    def __init__(self, window_size=30, num_features=21, bottleneck_dim=32):
        super().__init__()

        self.window_size = window_size
        self.num_features = num_features
        self.input_dim = window_size * num_features

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x_flat = x.view(batch_size, -1)

        encoded = self.encoder(x_flat)
        reconstructed = self.decoder(encoded)

        reconstructed = reconstructed.view(
            batch_size,
            self.window_size,
            self.num_features
        )

        return reconstructed


if __name__ == "__main__":
    model = LinearAutoencoder(
        window_size=30,
        num_features=21,
        bottleneck_dim=32
    )

    dummy_batch = torch.randn(64, 30, 21)

    output = model(dummy_batch)

    print("Linear Autoencoder created successfully")
    print("Input shape:", dummy_batch.shape)
    print("Output shape:", output.shape)