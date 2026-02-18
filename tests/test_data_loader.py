import pandas as pd
import numpy as np
import os

from src.data_loader import (
    load_data,
    get_tsla_close,
    get_spy_close,
    get_bnd_close
)


def test_load_data():

    os.makedirs("data/processed", exist_ok=True)

    df = pd.DataFrame({
        "TSLA": [100, 101, 102],
        "SPY": [400, 401, 402],
        "BND": [80, 81, 82]
    })

    df.to_csv("data/processed/historical_prices.csv")

    loaded = load_data()

    assert isinstance(loaded, pd.DataFrame)
    assert "TSLA" in loaded.columns


def test_get_tsla_close():

    df = pd.DataFrame({
        "TSLA": [100, 101, 102]
    })

    tsla = get_tsla_close(df)

    assert len(tsla) == 3


def test_get_spy_close():

    df = pd.DataFrame({
        "SPY": [400, 401, 402]
    })

    spy = get_spy_close(df)

    assert len(spy) == 3


def test_get_bnd_close():

    df = pd.DataFrame({
        "BND": [80, 81, 82]
    })

    bnd = get_bnd_close(df)

    assert len(bnd) == 3
