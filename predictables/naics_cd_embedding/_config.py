"""Configuration for the NAICS code embedding model."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NAICSSingleLevelConfig:
    """Configuration for a single level of NAICS codes.

    Attributes
    ----------
    nunique : int
        The number of unique NAICS codes at this level.
    embed_dim : int
        The size of the embedding vectors for each NAICS code.
    dropout : float
        The dropout probability for the model.
    """

    nunique: int
    embed_dim: int
    dropout: float
    # lambda_: float  # noqa: ERA001
    # l1_ratio: float  # noqa: ERA001

    def __post_init__(self):
        """Ensure that the configuration is valid."""
        assert self.nunique is not None, "nunique must be provided"
        assert self.embed_dim is not None, "embed_dim must be provided"
        assert self.dropout is not None, "dropout must be provided"
        # assert self.lambda_ is not None, "lambda must be provided"
        # assert self.l1_ratio is not None, "l1_ratio must be provided"

        assert (
            self.nunique > 0
        ), f"nunique must be a positive integer, got {self.nunique}"
        assert (
            self.embed_dim > 0
        ), f"embed_dim must be a positive integer, got {self.embed_dim}"
        assert (
            0 <= self.dropout < 1
        ), f"dropout must be in the range [0, 1), got {self.dropout}"


@dataclass
class NAICSConfig:
    """Global configuration for the NAICS embedding model.

    Attributes
    ----------
    naics2 : NAICSSingleLevelConfig
        Configuration for the 2-digit NAICS codes.
    naics3 : NAICSSingleLevelConfig
        Configuration for the 3-digit NAICS codes.
    naics4 : NAICSSingleLevelConfig
        Configuration for the 4-digit NAICS codes.
    naics5 : NAICSSingleLevelConfig
        Configuration for the 5-digit NAICS codes.
    naics6 : NAICSSingleLevelConfig
        Configuration for the 6-digit NAICS codes.
    is_classification : bool
        Whether the model is used for classification or regression.
    lambda_ : float
        The regularization penalty.
    l1_ratio : float
                The proportion of the regularization penalty applied to the L1 norm.
    """

    is_classification: bool = False
    current_level: int = 2
    naics2: NAICSSingleLevelConfig | None = None
    naics3: NAICSSingleLevelConfig | None = None
    naics4: NAICSSingleLevelConfig | None = None
    naics5: NAICSSingleLevelConfig | None = None
    naics6: NAICSSingleLevelConfig | None = None

    def __post_init__(self):
        """Ensure that the configuration is valid."""
        if self.naics2 is None:
            self.current_level = 2
        elif self.naics3 is None:
            self.current_level = 3
        elif self.naics4 is None:
            self.current_level = 4
        elif self.naics5 is None:
            self.current_level = 5
        elif self.naics6 is None:
            self.current_level = 6

        # assert (
        #     self.lambda_ > 0
        # ), f"lambda_ must be larger than 0, but {self.lambda_} was provided."
        # assert (
        #     0 <= self.l1_ratio <= 1   # noqa: ERA001
        # ), f"l1_ratio must be between 0 and 1, but got {self.l1_ratio}."
        # assert (
        #     0 <= self.l1_ratio <= 1   # noqa: ERA001
        # ), f"l1_ratio must be between 0 and 1, but got {self.l1_ratio}."

    def add(
        self,
        level: int | None = None,
        nunique: int | None = None,
        embed_dim: int | None = None,
        dropout: float | None = None,
    ) -> None:
        """Add a configuration for a specific NAICS code level.

        Parameters
        ----------
        level : int, optional
            The NAICS code level to configure.
        nunique : int, required
            The number of unique NAICS codes at this level.
        embed_dim : int, required
            The size of the embedding vectors for each NAICS code.
        dropout : float, required
            The dropout probability for the model.

        Raises
        ------
        ValueError
            If the NAICS level is invalid.
        """
        level_ = self.current_level if level is None else level

        # Validate the NAICS level
        if (level_ < 2) or (level_ > 6):
            raise ValueError(f"Invalid NAICS level: {level_}")

        # Create a configuration for a single level
        config_ = NAICSSingleLevelConfig(
            nunique=nunique,  # type: ignore
            embed_dim=embed_dim,  # type: ignore
            dropout=dropout,  # type: ignore
        )

        # Update the configuration for the specified level
        setattr(self, f"naics{level_}", config_)

    def get(self, level: int, attribute: str) -> int | float:
        """Get a specific attribute for a given NAICS code level.

        Parameters
        ----------
        level : int
            The NAICS code level to query.
        attribute : str
            The attribute to retrieve.

        Returns
        -------
        int | float
            The value of the specified attribute for the given NAICS code level.
        """
        return getattr(getattr(self, f"naics{level}"), attribute)  # type: ignore
