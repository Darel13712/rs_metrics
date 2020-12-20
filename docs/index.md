# Welcome to rs_metrics

This package implements metrics common in recommendation systems.

All metrics are paralleled by users.

## Installation

```bash
pip install rs_metrics
```

## Data Format

Metrics expect to get a `dict` mapping `user_id` with list of `item_id`.

!!! example
    ```python
    pred = {1: [1, 2, 3], 2: [3, 2, 4]}
    ```

You can also pass `pandas.DataFrame`, which will be converted `dict` to automatically with 
`convert_to_pandas` function. 

Default columns are `user_col='user_id', item_col='item_id'`.