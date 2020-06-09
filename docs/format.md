# Data Format

Metrics expect to get a `dict` mapping `user_id` with list of `item_id`.

!!! example
    ```python
    pred = {1: [1, 2, 3], 2: [3, 2, 4]}
    ```

If you have recommendations in `pandas.DataFrame`, you can convert them to this format using 
`rs_metrics.pandas_to_dict`

```python
def pandas_to_dict(df, user_col='user_id', item_col='item_id'):
    return df.groupby(user_col)[item_col].agg(list).to_dict()
```