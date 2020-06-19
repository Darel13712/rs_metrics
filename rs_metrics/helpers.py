def pandas_to_dict(df, user_col='user_id', item_col='item_id'):
    return df.groupby(user_col)[item_col].agg(list).to_dict()


def flatten_list(arr):
    return [item for sublist in arr for item in sublist]
