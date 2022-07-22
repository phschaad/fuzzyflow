from typing import Tuple
import numpy as np

def generate_dataset(
    n: int, tiny_percentage: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tiny = round((tiny_percentage / 100) * n)
    n_regular = n - n_tiny
    tiny_sample = np.random.uniform(low=1e-30, high=1e-10, size=(n_tiny,))
    large_sample = np.random.uniform(low=1e0, high=1e10, size=(n_regular,))
    sorted_data = np.concatenate((tiny_sample, large_sample))
    unsorted_data = sorted_data.copy()
    np.random.shuffle(unsorted_data)
    reversed_data = sorted_data[::-1]
    return (sorted_data, unsorted_data, reversed_data)


def generate_split_set(
    n: int, tiny_percentage: float,
    tiny_low=1e-30, tiny_high=1e-10,
    regular_low=1e0, regular_high=1e10
) -> np.ndarray:
    n_tiny = round((tiny_percentage / 100) * n)
    n_regular = n - n_tiny
    tiny_sample = np.random.default_rng().uniform(
        tiny_low, tiny_high, n_tiny
    )
    large_sample = np.random.default_rng().uniform(
        regular_low, regular_high, n_regular
    )
    conc = np.concatenate((tiny_sample, large_sample))
    np.random.shuffle(conc)
    return conc
