import numpy as np
import torch
import typing


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: int = 0,
        path: str = "checkpoint.pt",
        trace_func: typing.Callable = print,
    ):
        """Initialize the early stopping object.

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
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
