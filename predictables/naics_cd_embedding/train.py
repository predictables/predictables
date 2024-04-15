"""Train an embedding model for NAICS codes."""

import torch
from ._defaults import NAICSDefaults
from ._data import NAICSDataLoader
from ._early_stopper import NAICSEarlyStopper
from predictables.config import logger


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
            logger.info(f"Starting Epoch {epoch}")

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
                    logger.error(f"Error during training/validation: {e}")
                    continue

                early_stopper(val_loss, model)
                if early_stopper.early_stop:
                    logger.info(f"Early stopping triggered after epoch {epoch}")
                    break

            logger.info(
                f"Epoch {epoch} completed | Training Loss: {train_loss.item():.4f} | Validation Loss: {val_loss.item():.4f}"
            )
            if early_stopper.early_stop:
                break
    except Exception as e:
        logger.critical(f"Failed to complete training loop: {e}")
