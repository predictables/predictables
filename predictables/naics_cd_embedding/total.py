from __future__ import annotations
from dataclasses import dataclass
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from sklearn.preprocessing import OneHotEncoder  # type: ignore

"""Train an embedding model for NAICS codes."""

import torch
from _defaults import NAICSDefaults
from _data import NAICSDataLoader
from _early_stopper import NAICSEarlyStopper

import numpy as np
import typing
import os
from dotenv import load_dotenv

load_dotenv()

DATA_FILENAME = "final_naics_data.parquet"
TARGET_COLUMN_NAME = "target"
BATCH_SIZE = 32


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
        self.embedding_2_digit = nn.Embedding(config.get(2, "nunique"), config.get(2, "embed_dim"))
        self.embedding_3_digit = nn.Embedding(config.get(3, "nunique"), config.get(3, "embed_dim"))
        self.embedding_4_digit = nn.Embedding(config.get(4, "nunique"), config.get(4, "embed_dim"))
        self.embedding_5_digit = nn.Embedding(config.get(5, "nunique"), config.get(5, "embed_dim"))
        self.embedding_6_digit = nn.Embedding(config.get(6, "nunique"),config.get(6, "embed_dim"),
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


@dataclass
class NAICSDefaults:
    """Default values for the NAICS embedding model."""

    _config: NAICSConfig | None = None
    _model: NAICSEmbeddingModel | None = None
    _loss: torch.nn.BCEWithLogitsLoss | torch.nn.MSELoss | None = None
    _optimizer: torch.optim.Adam | None = None

    def __post_init__(self):
        """Initialize the default values for the NAICS embedding model."""
        self.config()
        self.model()
        self.loss()
        self.optim()

    def config(self) -> None:
        """Get a default configuration for the NAICS embedding model."""
        config = NAICSConfig(is_classification=True)  # type: ignore
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
            if self._config.is_classification  # type: ignore
            else torch.nn.MSELoss()
        )

    def optim(self) -> None:
        """Get the default optimizer for the NAICS embedding model."""
        self._optim = torch.optim.Adam(self._model.parameters(), lr=0.001)  # type: ignore

    def get(
        self,
    ) -> tuple[NAICSConfig, NAICSEmbeddingModel, torch.nn.Module, torch.optim.Adam]:
        """Get the default values for the NAICS embedding model."""
        return self._config, self._model, self._loss, self._optim  # type: ignore


def training_loop() -> None:
    """Train the NAICS embedding model."""
    try:
        config, model, loss, optim = NAICSDefaults().get()
        data_loader = NAICSDataLoader()
        early_stopper = NAICSEarlyStopper()

        # Train the model
        epoch = 0
        while True:
            epoch += 1
            print(f"Starting Epoch {epoch}")
            # logger.info(f"Starting Epoch {epoch}")

            for train_X, train_y, val_X, val_y in data_loader:
                optim.zero_grad()
                try:
                    train_probs = model(*train_X)
                    train_loss = loss(train_probs.squeeze(), train_y)
                    train_loss.backward()
                    optim.step()

                    with torch.no_grad():
                        val_probs = model(*val_X)
                        val_loss = loss(val_probs.squeeze(), val_y)
                except Exception as e:
                    print(f"Error during training/validation: {e}")
                    # logger.error(f"Error during training/validation: {e}")
                    continue

                early_stopper(val_loss, model)
                if early_stopper.early_stop:
                    print(f"Early stopping triggered after epoch {epoch}")
                    # logger.info(f"Early stopping triggered after epoch {epoch}")
                    break

            print(
                f"Epoch {epoch} completed | Training Loss: {train_loss.item():.4f} | Validation Loss: {val_loss.item():.4f}"
            )
            # logger.info(
            #     f"Epoch {epoch} completed | Training Loss: {train_loss.item():.4f} | Validation Loss: {val_loss.item():.4f}"
            # )
            if early_stopper.early_stop:
                break
    except Exception as e:
        print(f"Failed to complete training loop: {e}")
        # logger.critical(f"Failed to complete training loop: {e}")




defaults = {
    "patience": int(os.getenv("EARLY_STOPPER_PATIENCE", "7"))
    if os.getenv("EARLY_STOPPER_PATIENCE")
    else 7,
    "verbose": bool(os.getenv("EARLY_STOPPER_VERBOSE", "False"))
    if os.getenv("EARLY_STOPPER_VERBOSE")
    else False,
    "delta": float(os.getenv("EARLY_STOPPER_DELTA", "0"))
    if os.getenv("EARLY_STOPPER_DELTA")
    else 0,
    "path": os.getenv("EARLY_STOPPER_CHECKPOINT_PATH", "checkpoint.pt")
    if os.getenv("EARLY_STOPPER_CHECKPOINT_PATH")
    else "checkpoint.pt",
}


class NAICSEarlyStopper:
    """Stop training early if the model is not improving.

    The early stopper does the following:
    1. Monitors the state of the model
    2. Saves the model whenever validation loss decreases
    3. Stops training if validation loss doesn't improve after
       a given patience.

    Here 'patience' is the number of training epochs we are
    willing to wait for the validation loss to further improve.

    The early stopper settings can be configured by setting
    the following environment variables:
    - EARLY_STOPPER_PATIENCE: How long to wait after last time
      validation loss improved.
    - EARLY_STOPPER_VERBOSE: If True, prints a message for each
        validation loss improvement.
    - EARLY_STOPPER_DELTA: Minimum change in the monitored quantity
        to qualify as an improvement.
    - EARLY_STOPPER_CHECKPOINT_PATH: Path for the checkpoint to be
        saved to.
    """

    def __init__(
        self,
        patience: int | None = None,
        verbose: bool | None = None,
        delta: int | None = None,
        path: str | None = None,
        trace_func: typing.Callable = print,
    ):
        """Initialize the early stopper.

        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss
            improved, by default 7
        verbose : bool, optional
            If True, prints a message for each validation loss
            improvement, by default False
        delta : float, optional
            Minimum change in the monitored quantity to qualify
            as an improvement, by default 0
        path : str, optional
            Path for the checkpoint to be saved to, by default
            'checkpoint.pt'
        trace_func : function, optional
            Trace print function, by default print
        """
        self.patience = defaults["patience"] if patience is None else patience
        self.verbose = defaults["verbose"] if verbose is None else verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_score = np.inf
        self.delta = defaults["delta"] if delta is None else delta
        self.path = defaults["path"] if path is None else path
        self.trace_func = trace_func
        self.num_almost_triggered = 0

        print(
            f"Initialized EarlyStopper with patience={self.patience}, verbose={self.verbose}, delta={self.delta}, path={self.path}"
        )
        # logger.debug(
        #     f"Initialized EarlyStopper with patience={self.patience}, verbose={self.verbose}, delta={self.delta}, path={self.path}"
        # )

    def __call__(self, val_score: float, model: torch.nn.Module) -> None:
        """Inspect the current state of the model.

        This method is called at the end of each epoch to
        determine if the model should be saved or if training
        should be stopped.

        The NAICSEarlyStopper object maintains a record of:
        1. The best validation score
        2. The current number of epochs since the best score
           was last updated

        If the validation score improves, the model is saved, the
        best score is updated, and the counter is reset to 0.

        If the validation score does not improve, the counter is
        incremented. If the counter exceeds the patience attribute
        of the class, the early_stop attribute is set to True, and
        training is stopped.

        Here 'improvement' may optionally be constrained to be
        an increase in the validation score in excess of the
        `delta` (>= 0) attribute of the class. `delta` defaults
        to 0, and is another parameter that can be set before
        training.

        The improvement is determined by the user's choice of
        metric, and is not dependent on any particular validation
        score metric. The chosen metric is calculed during training
        and passed to this method.

        Note also that `val_score` is a *score*, not a *loss*, and
        so the expectation is that the score will increase with
        better performance. If you are tracking a loss, you can
        simply negate the loss before passing it to this method.

        Parameters
        ----------
        val_score : float
            The validation score to monitor. Note that the particular
            metric to monitor is not specified here -- it is calculated
            during training and passed to this method.
        model : torch.nn.Module
            The model to save if the validation loss improves.
        """
        # Initialize the best score if it is None and checkpoint
        # the model
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)

        # If the passed validation score is less than the best
        # score plus the delta, increment the counter.
        elif val_score > self.best_score - self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter incremented: counter/patience = {self.counter}/{self.patience}"
            )
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            # logger.debug(
            #     f"EarlyStopping counter incremented: counter/patience = {self.counter}/{self.patience}"
            # )
            # self.trace_func(
            #     f"EarlyStopping counter: {self.counter} out of {self.patience}"
            # )
            if self.counter >= self.patience:
                print("Early stopping triggered, stopping training.")
                # logger.info("Early stopping triggered, stopping training.")

                self.early_stop = True

            elif self.counter - 1 == self.patience:
                self.num_almost_triggered += 1
                print(
                    f"After one more epoch, early stopping will be triggered if validation score does not improve. This has now happened {self.num_almost_triggered} time(s)."
                )
                # logger.info(
                #     f"After one more epoch, early stopping will be triggered if validation score does not improve. This has now happened {self.num_almost_triggered} time(s)."
                # )

        else:
            print(
                f"Validation score improved significantly ({val_score} > {self.best_score}); resetting early stopping counter (currently at {self.counter})."
            )
            # logger.debug(
            #     f"Validation score improved significantly ({val_score} > {self.best_score}); resetting early stopping counter (currently at {self.counter})."
            # )
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score: float, model: torch.nn.Module) -> None:
        """Save model when validation loss improves."""
        if self.verbose:
            self.trace_func(
                f"Validation score improved ({self.best_val_score:.6f} --> {val_score:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.best_val_score = val_score
        print(
            f"Checkpoint saved: Improved score to {val_score}, model saved to {self.path}"
        )
        # logger.info(
        #     f"Checkpoint saved: Improved score to {val_score}, model saved to {self.path}"
        # )
