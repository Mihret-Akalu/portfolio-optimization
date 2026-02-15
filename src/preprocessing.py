def split_time_series(series):

    train = series.loc[: "2024-12-31"]

    test = series.loc["2025-01-01":]

    return train, test
