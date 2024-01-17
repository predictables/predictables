import polars as pl


class SingularValueDecomposition:
    def __init__(self, A: pl.LazyFrame):
        # Matrix A
        self.A = A  # m x n matrix

        # Dimensions of A
        self.n_rows = A.shape[0]
        self.n_cols = A.shape[1]
        self.m, self.n = self.n_rows, self.n_cols

        # Singular values of A
        self.U = None  # m x m orthogonal matrix
        self.S = (
            None  # m x n diagonal matrix with non-negative real numbers on the diagonal
        )
        self.V = None  # n x n orthogonal matrix

    def subspace_iteration(self):
        """
        Perform subspace iteration to find the singular values and vectors of A.
        """
        # Initial values
        V = pl.DataFrame(
            pl.Series([1] + ([0] * (self.n_rows - 1))).cast(pl.Float32).alias("Col0")
        )

        V = V.with_columns(
            [
                pl.Series(([0] * i) + [1] + ([0] * (self.n_rows - i - 1)))
                .cast(pl.Float32)
                .alias(f"Col{i}")
                for i in range(1, self.n_rows)
            ]
        )

        # Iterate
