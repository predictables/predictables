# import numpy as np
# import pytest

# from PredicTables.util.stats import kl_divergence

# @pytest.fixture
# def valid():
#     return [1, 2, 3]

# @pytest.fixture
# def invalid():
#     return [1, 2, 3]

# @pytest.fixture
# def empty():
#     return []

# @pytest.fixture
# def zero():
#     return [0]

# @pytest.fixture
# def negative():
#     return [-1]

# @pytest.fixture
# def nan():
#     return [np.nan]

# @pytest.fixture
# def inf():
#     return [np.inf]

# def test_kl_divergence_valid(valid):
#     assert (
#         kl_divergence(valid, valid) == 0
#     ), f"kl_divergence({valid}, {valid}) should be 0"

# def test_kl_divergence_invalid(invalid):
#     with pytest.raises(ValueError):
#         kl_divergence(invalid, invalid)
