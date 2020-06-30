def item_pop(df, user_col='user_id', item_col='item_id'):
    res = df.groupby(item_col)[user_col].nunique().sort_values(ascending=False)
    return res / df[user_col].nunique()
