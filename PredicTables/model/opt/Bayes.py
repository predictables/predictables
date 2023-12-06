from .src.bayes import (
    get_space,
    parallel_optimization,
    get_model_obj,
    _objective_function_no_pruning,
)

from PredicTables.util import model_object_to_model_name


class Bayes:
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        model=None,
        model_name: str = None,
        max_iter: int = 100,
        task_type: str = "classification",
        optimizer=None,
        objective_func=None,
        batch_size: int = 4,
        **kwargs,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.task_type = task_type
        self.batch_size = batch_size

        # Only need to pass one of model_name or model (object)
        if model_name is None and model is None:
            # but do need to pass one of them
            raise ValueError("Either model_name or model must be provided.")
        elif model_name is not None and model is None:
            # model name -> model object
            self.model = get_model_obj(model_name, task_type)

        if model_name is None and model is not None:
            # model object -> model name
            self.model_name = model_object_to_model_name(model)
        else:
            self.model_name = model_name

        # Default space is determined by model_name
        self.space = get_space(self.model_name, **kwargs)
        self.max_iter = max_iter

        # Default optimizer is skopt.gp_minimize
        if optimizer is None:
            self.optimizer = parallel_optimization
        else:
            self.optimizer = optimizer

        # Default objective function is _objective_function_no_pruning
        if objective_func is None:
            self.objective_func = _objective_function_no_pruning
        else:
            self.objective_func = objective_func

    def set_space(self, **kwargs):
        """
        Updates the space used for hyperparameter optimization from the
        default space. Keyword arguments are passed to the default space
        function to update the default space values.
        """
        self.space = get_space(self.model_name, **kwargs)

    def set_model(self, model=None, model_name=None):
        """
        Not sure whether or not this is a good idea or needed. Can pass a string
        representing the model name or a model object. Updates the self.model
        attribute and self.model_name attribute.
        """
        if model_name is None and model is None:
            # but do need to pass one of them
            raise ValueError("Either model_name or model must be provided.")
        elif model_name is not None and model is None:
            # model name -> model object
            self.model = get_model_obj(model_name, self.task_type)

        if model_name is None and model is not None:
            # model object -> model name
            self.model_name = model_object_to_model_name(model)
        else:
            self.model_name = model_name

    def set_objective(self, objective_func):
        """
        Updates the objective function used for hyperparameter optimization.
        """
        self.objective_func = objective_func

    def set_optimizer(self, optimizer):
        """
        Updates the optimizer used for hyperparameter optimization.
        """
        self.optimizer = optimizer

    def optimize(self, **kwargs):
        """
        Runs hyperparameter optimization using the optimizer specified in
        self.optimizer. Keyword arguments are passed to the optimizer.
        """
        return self.optimizer(
            objective_func=_objective_function_no_pruning,
            space=self.space,
            n_calls=self.max_iter,
            batch_size=self.batch_size,
            **kwargs,
        )
