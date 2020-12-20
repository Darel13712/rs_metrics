import pandas as pd


def pandas_to_dict(df, user_col='user_id', item_col='item_id'):
    return df.groupby(user_col)[item_col].agg(list).to_dict()


def flatten_list(arr):
    return [item for sublist in arr for item in sublist]


def convert_pandas(metric):
    def wrap(true, pred, k=10, user_col='user_id', item_col='item_id'):
        if type(true) is pd.DataFrame:
            true = pandas_to_dict(true, user_col, item_col)

        if type(pred) is pd.DataFrame:
            pred = pandas_to_dict(pred, user_col, item_col)

        return metric(true, pred, k)
    return wrap
