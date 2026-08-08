"""
utils.py - Shared helper utilities for the demand forecasting project.
"""

import logging
import pandas as pd
from pathlib import Path


def setup_logging(level=logging.INFO):
    """Configure root logger with a simple format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_data(data_path: Path):
    """
    Load the three CSV files and return raw DataFrames.

    Returns
    -------
    orders_train, orders_test, hub_metadata : pd.DataFrame
    """
    data_path = Path(data_path)

    orders_train  = pd.read_csv(data_path / "orders_train.csv",  parse_dates=["Date"])
    orders_test   = pd.read_csv(data_path / "orders_test.csv",   parse_dates=["Date"])
    hub_metadata  = pd.read_csv(data_path / "hub_metadata.csv")

    print(f"[load_data] orders_train  : {orders_train.shape}")
    print(f"[load_data] orders_test   : {orders_test.shape}")
    print(f"[load_data] hub_metadata  : {hub_metadata.shape}")

    return orders_train, orders_test, hub_metadata
