""" Load dataset """

import datasets


def get_dataset(name):
    """Load and return a dataset by name."""

    df = datasets.load_dataset(name, trust_remote_code=True)
    return df
