# Data Format

Metrics expect to get a `dict` mapping `user_id` with list of `item_id`.

!!! example
    ```python
    pred = {1: [1, 2, 3], 2: [3, 2, 4]}
    ```

If you have recommendations in `pandas.DataFrame`, you can convert them to this format with

```python
df.groupby('user_id')['item_id'].agg(list).to_dict()
```