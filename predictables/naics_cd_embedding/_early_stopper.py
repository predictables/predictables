import numpy as np
import torch
import typing


class NAICSEarlyStopper:
    """Stop training early if the model is not improving.

    The early stopper does the following:
    1. Monitors the state of the model
    2. Saves the model whenever validation loss decreases
    3. Stops training if validation loss doesn't improve after
       a given patience.

    Here 'patience' is the number of training epochs we are
    willing to wait for the validation loss to further improve.
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: int = 0,
        path: str = "checkpoint.pt",
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
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

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
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score: float, model: torch.nn.Module) -> None:
        """Save model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_score_min = val_score
