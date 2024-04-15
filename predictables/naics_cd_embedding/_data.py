from __future__ import annotations
from sklearn.preprocessing import OneHotEncoder  # type: ignore
import torch  # type: ignore
import polars as pl

DATA_FILENAME = "final_naics_data.parquet"
TARGET_COLUMN_NAME = "target"
BATCH_SIZE = 32


class NAICSDataset(torch.utils.data.Dataset):
    """Custom Dataset for handling NAICS data with one-hot encoding of categorical NAICS codes."""

    def __init__(self, lf: pl.LazyFrame):
        self.encoder = OneHotEncoder(sparse_output=False)

        # Validate that lf contains columns named 'naics_2_cd' through 'naics_6_cd'
        for i in range(2, 6):
            assert (
                f"naics_{i}_cd" in lf.columns
            ), f"Column 'naics_{i}_cd' not found in DataFrame. Expected a column with this exact name."

        assert (
            ("naics_6_cd" in lf.columns) | ("naics_cd" in lf.columns)
        ), "Column 'naics_6_cd' or 'naics_cd' not found in DataFrame. Expected a column with one of these exact names."

        # Assuming 'naics_2_cd' through 'naics_6_cd' need encoding
        naics_features = (
            lf.select(
                [
                    pl.col("naics_2_cd"),
                    pl.col("naics_3_cd"),
                    pl.col("naics_4_cd"),
                    pl.col("naics_5_cd"),
                    pl.col("naics_6_cd"),
                ]
            )
            .collect()
            .to_numpy()
        )
        encoded_features = self.encoder.fit_transform(naics_features)

        # Convert encoded features to tensor
        self.features = torch.tensor(encoded_features, dtype=torch.float32)

        # Assuming 'target' as labels
        self.labels = torch.tensor(
            lf.select(TARGET_COLUMN_NAME).collect().to_numpy(), dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class NAICSDataLoader(torch.utils.data.DataLoader):
    """A data loader for the NAICS embedding model."""

    def __init__(
        self, lf: pl.LazyFrame, batch_size: int = BATCH_SIZE, shuffle: bool = True
    ):
        # Convert the df into a Dataset
        dataset = NAICSDataset(lf)
        super(NAICSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
