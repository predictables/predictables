"""Define the default configuration for the NAICS code embedding model."""

from __future__ import annotations
from dataclasses import dataclass
from predictables.naics_cd_embedding_model import NAICSEmbeddingModel
from predictables.naics_cd_embedding_config import NAICSConfig
import torch


@dataclass
class NAICSDefaults:
    """Default values for the NAICS embedding model."""

    _config: NAICSConfig = None
    _model: NAICSEmbeddingModel = None
    _loss: torch.nn.BCEWithLogitsLoss | torch.nn.MSELoss = None
    _optimizer: torch.optim.Adam = None

    def __post_init__(self):
        """Initialize the default values for the NAICS embedding model."""
        self.config()
        self.model()
        self.loss()
        self.optim()

    def config(self) -> None:
        """Get a default configuration for the NAICS embedding model."""
        config = NAICSConfig(is_classification=True)
        config.add(2, 20, 128, 0.5)
        config.add(3, 100, 128, 0.4)
        config.add(4, 500, 128, 0.3)
        config.add(5, 1000, 128, 0.2)
        config.add(6, 2000, 128, 0.1)
        self._config = config

    def model(self) -> None:
        """Get the default NAICS embedding model."""
        self._model = NAICSEmbeddingModel(self._config)

    def loss(self) -> None:
        """Get the default loss function for the NAICS embedding model."""
        self._loss = (
            torch.nn.BCEWithLogitsLoss()
            if self._config.is_classification
            else torch.nn.MSELoss()
        )

    def optim(self) -> None:
        """Get the default optimizer for the NAICS embedding model."""
        self._optim = torch.optim.Adam(self._model.parameters(), lr=0.001)

    def get(
        self,
    ) -> tuple[NAICSConfig, NAICSEmbeddingModel, torch.nn.Module, torch.optim.Adam]:
        """Get the default values for the NAICS embedding model."""
        return self._config, self._model, self._loss, self._optim
