"""Define a PyTorch model that generates an embedding of NAICS codes in a lower-dimensional space.

The model leverages the hierarchical structure of NAICS codes to learn a lower-dimensional representation of the codes.

NAICS Code Structure
--------------------
- 2-digit: Sector (eg 23 is Construction)
- 3-digit: Subsector (eg 236 is Construction of Buildings) is a subset of a sector
- 4-digit: Industry Group (eg 2361 is Residential Building Construction) is a subset of a subsector
- 5-digit: Industry (eg 23611 is Residential Building Construction) is a subset of an industry group
- 6-digit: National Industry (eg 236118 is Residential Building Construction) is a subset of an industry

Basic Architecture
------------------
1. Generate an embedding for 2-digit NAICS codes, tuned on the actual target variable
2. Generate a delta embedding for 3-digit NAICS codes, tuned on the residuals of the 2-digit model
3. Generate a delta embedding for 4-digit NAICS codes, tuned on the residuals of the 3-digit model
4. Generate a delta embedding for 5-digit NAICS codes, tuned on the residuals of the 4-digit model
5. Generate a delta embedding for 6-digit NAICS codes, tuned on the residuals of the 5-digit model
6. Each delta embedding is added to the embedding of the previous level to generate the final embedding

Model Architecture
------------------
1. Generate an initial embedding for 2-digit NAICS codes, tuned on the actual target variable.
2. For each subsequent level (3-digit to 6-digit), generate a delta embedding tuned on the residuals of the model at the previous level.
  - eg 3-digit delta embedding is tuned on the residuals of the 2-digit model, 4-digit delta embedding is tuned on the residuals of the 3-digit model, etc.
3. Each delta embedding is added to the embedding of the previous level to generate the final embedding for each NAICS code level.
  - eg 3-digit embedding = 2-digit embedding + 3-digit delta embedding, 4-digit embedding = 3-digit embedding + 4-digit delta embedding, etc.

Inputs
------
- A batch of NAICS codes at each level, formatted as tensors.

Outputs
-------
- A tensor containing the embeddings for each NAICS code in the batch.

Dependencies
------------
- PyTorch 2.x

Example
-------
```python
>>> model = NAICSEmbeddingModel(embed_dim=128)
>>> data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
>>> for inputs, targets in data_loader:
...     predictions = model(inputs)
...     loss = compute_loss(predictions, targets)
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()
```
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from predictables.naics_cd_embedding._config import NAICSConfig


class NAICSEmbeddingModel(nn.Module):
    """Generate an embedding of NAICS codes in a lower-dimensional space.

    The model leverages the hierarchical structure of NAICS codes to learn a lower-dimensional representation of the codes.

    Methods
    -------
    forward(naics_codes)
        Generate embeddings for a batch of NAICS codes at each level.
    """

    def __init__(self, config: NAICSConfig):
        super(NAICSEmbeddingModel, self).__init__()
        self.config = config

        # Embedding layers for each NAICS code level
        self.embedding_2_digit = nn.Embedding(
            config.get(2, "nunique"),  # type: ignore
            config.get(2, "embed_dim"),  # type: ignore
        )
        self.embedding_3_digit = nn.Embedding(
            config.get(3, "nunique"),  # type: ignore
            config.get(3, "embed_dim"),  # type: ignore
        )
        self.embedding_4_digit = nn.Embedding(
            config.get(4, "nunique"),  # type: ignore
            config.get(4, "embed_dim"),  # type: ignore
        )
        self.embedding_5_digit = nn.Embedding(
            config.get(5, "nunique"),  # type: ignore
            config.get(5, "embed_dim"),  # type: ignore
        )
        self.embedding_6_digit = nn.Embedding(
            config.get(6, "nunique"),  # type: ignore
            config.get(6, "embed_dim"),  # type: ignore
        )

        # Delta embedding layers for each NAICS code level
        self.delta_embedding_3_digit = nn.Embedding(
            config.get(3, "nunique"),  # type: ignore
            config.get(3, "embed_dim"),  # type: ignore
        )
        self.delta_embedding_4_digit = nn.Embedding(
            config.get(4, "nunique"),  # type: ignore
            config.get(4, "embed_dim"),  # type: ignore
        )
        self.delta_embedding_5_digit = nn.Embedding(
            config.get(5, "nunique"),  # type: ignore
            config.get(5, "embed_dim"),  # type: ignore
        )
        self.delta_embedding_6_digit = nn.Embedding(
            config.get(6, "nunique"),  # type: ignore
            config.get(6, "embed_dim"),  # type: ignore
        )

        # Final linear layer for prediction
        self.linear = nn.Linear(config.get(6, "embed_dim"), 1)  # type: ignore

    def forward(
        self,
        naics_2_digit: torch.Tensor,
        naics_3_digit: torch.Tensor,
        naics_4_digit: torch.Tensor,
        naics_5_digit: torch.Tensor,
        naics_6_digit: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the model embeddings for a batch of NAICS codes at each level.

        The steps in the forward pass are as follows:
        1. Take a tensor of 2-digit NAICS codes and generate embeddings based on the actual target variable.
        2. Calculate the residuals between the embeddings and the target variable.
        3. Take a tensor of 3-digit NAICS codes and generate delta embeddings based on the residuals from the 2-digit model.
        4. Add the delta embeddings to the 2-digit embeddings to generate the final 3-digit embeddings.
        5. Repeat steps 2-4 for 4-digit to 6-digit NAICS codes.

        At each stage, and in order to minimize the risk of overfitting on a relatively small dataset, the model uses a dropout layer with a dropout probability of `self.dropout` (meaning that only `100 * self.dropout`% of the neurons are kept active during training).

        In order to ensure that the embeddings are not too large, the model also applies a normalization step to the embeddings at each level.

        Parameters
        ----------
        naics_2_digit : torch.Tensor
            A tensor of 2-digit NAICS codes.
        naics_3_digit : torch.Tensor
            A tensor of 3-digit NAICS codes.
        naics_4_digit : torch.Tensor
            A tensor of 4-digit NAICS codes.
        naics_5_digit : torch.Tensor
            A tensor of 5-digit NAICS codes.
        naics_6_digit : torch.Tensor
            A tensor of 6-digit NAICS codes.

        Returns
        -------
        torch.Tensor
            A tensor containing the probabilities for each NAICS code in the batch as predicted by the model.
        """
        # Generate embeddings for 2-digit NAICS codes
        embedding_2_digit = self.embedding_2_digit(naics_2_digit)
        embedding_2_digit = F.dropout(
            embedding_2_digit,
            p=self.config.get(2, "dropout"),
            training=self.training,  # type: ignore
        )
        embedding_2_digit = F.normalize(embedding_2_digit, p=2, dim=-1)

        # Generate delta embeddings and final embeddings for 3-digit NAICS codes
        delta_embedding_3_digit = self.delta_embedding_3_digit(naics_3_digit)
        embedding_3_digit = embedding_2_digit + delta_embedding_3_digit
        embedding_3_digit = F.dropout(
            embedding_3_digit,
            p=self.config.get(3, "dropout"),
            training=self.training,  # type: ignore
        )
        embedding_3_digit = F.normalize(embedding_3_digit, p=2, dim=-1)

        # Generate delta embeddings and final embeddings for 4-digit NAICS codes
        delta_embedding_4_digit = self.delta_embedding_4_digit(naics_4_digit)
        embedding_4_digit = embedding_3_digit + delta_embedding_4_digit
        embedding_4_digit = F.dropout(
            embedding_4_digit,
            p=self.config.get(4, "dropout"),
            training=self.training,  # type: ignore
        )
        embedding_4_digit = F.normalize(embedding_4_digit, p=2, dim=-1)

        # Generate delta embeddings and final embeddings for 5-digit NAICS codes
        delta_embedding_5_digit = self.delta_embedding_5_digit(naics_5_digit)
        embedding_5_digit = embedding_4_digit + delta_embedding_5_digit
        embedding_5_digit = F.dropout(
            embedding_5_digit,
            p=self.config.get(5, "dropout"),
            training=self.training,  # type: ignore
        )
        embedding_5_digit = F.normalize(embedding_5_digit, p=2, dim=-1)

        # Generate delta embeddings and final embeddings for 6-digit NAICS codes
        delta_embedding_6_digit = self.delta_embedding_6_digit(naics_6_digit)
        embedding_6_digit = embedding_5_digit + delta_embedding_6_digit
        embedding_6_digit = F.dropout(
            embedding_6_digit,
            p=self.config.get(6, "dropout"),
            training=self.training,  # type: ignore
        )
        embedding_6_digit = F.normalize(embedding_6_digit, p=2, dim=-1)

        # Compute the final output predictions
        logits = self.linear(embedding_6_digit)

        # Use sigmoid for binary classification
        if self.config.is_classification:
            return torch.sigmoid(logits)

        # No activation for non-classification tasks
        return logits  # type: ignore
