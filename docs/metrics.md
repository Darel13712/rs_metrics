# Metrics

## Relevance 

### HitRate

`#!py def hitrate(true, pred, k=10, user_col='user_id', item_col='item_id'):`

Shows what percentage of users has at least one relevant recommendation in their list.

$$
    HitRate@k =  \frac {\sum \limits_{u \in U} \max \limits_{i \in L(u)}(rel(i, u))} {|U|}
$$

- $rel(i, u)$ equals 1 if item $i$ is relevant to user $u$.
- $L(u)$ is a recommendation list of length $k$ for user $u$.


### Precision

`#!py def precision(true, pred, k=10, user_col='user_id', item_col='item_id'):`

Shows what percentage of items in recommendations are relevant, on average.

$$
Precision@k = \frac{1}{|U|} \sum_{u \in U} \frac{|rel_u \cap rec_u|}{|rec_u|}
$$

- $rel_u$ – items relevant for user $u$
- $rec_u$ – top-k items recommended to user $u$

### Mean Average Precision

`#!py def mapr(true, pred, k=10, user_col='user_id', item_col='item_id'):`

$$
AP@k(u) = \frac{1}{|rel_u|} \sum_{i \in rec_u} rel(i, u)Precision@pos_{i, u}(u)
$$

$$
MAP@k = \frac{1}{|U|} \sum_{u \in U} AP@k(u)
$$

- $rel(i, u)$ – equals 1 if item $i$ is relevant to user $u$
- $rel_u$ – items relevant for user $u$
- $rec_u$ – top-k recommendations for user $u$
- $pos_{i, u}$ – position of item $i$ in recommendation list $rec_u$

### Recall

`#!py def recall(true, pred, k=10, user_col='user_id', item_col='item_id'):`

Shows what percentage of relevant items appeared in recommendations, on average.

$$
Recall@k = \frac{1}{|U|} \sum_{u \in U} \frac{|rel_u \cap rec_u|}{|rel_u|}
$$

- $rel_u$ – items relevant for user $u$
- $rec_u$ – top-k items recommended to user $u$

## Ranking
 
### NDCG
`#!py def ndcg(true, pred, k=10, user_col='user_id', item_col='item_id'):`

Normalized discounted cumulative gain is a ranking quality metric. 
It takes position of relevant items into account, 
promoting them to the beginning of the list.

Although relevance can be any function (star rating), 
simple binary variant is usually used:


$$
    DCG@k = \sum_{i=1}^k \frac{2^{rel(i)}-1} {log_2(i+1)}
$$

$$
    NDCG@k = \frac{DCG@k} {IDCG@k}
$$

- $rel(i)$ equals 1 if item at position $i$ is relevant, i.e `pred[u][i]` $\in$ `#!py true[u]`.
- $IDCG@k = \max(DCG@k)$

This value is averaged across all users.

### MRR

`#!py def mrr(true, pred, k=10, user_col='user_id', item_col='item_id'):`

Mean Reciprocal Rank shows inverted position of the first relevant item, on average.

## Diversity

### α-NDCG

`#!py def a_ndcg(true, pred, aspects, k=10, alpha=0.5, user_col='user_id', item_col='item_id'):`

- `aspects`: 
    dictionary which maps users to aspect list containing items for each aspect.
    
    Example: `#!py {1: [{1, 2, 3}, {3, 4}], 2: [{1}]}`
    

- `alpha` $\in [0,1]$:
   controls redundancy penalty. 
   The bigger the number the more metric penalizes items from the same aspect.
   
$α\text{-}NDCG$ is based on $NDCG$ but it is aspect and redundancy-aware, 
which makes it a measure of diversity:

$$
    \alpha \text{-}DCG@k=\sum_{i=1}^k \frac{1}{log_2(i + 1)}
    \sum_{a \in \mathcal{A}} rel(i|a) \prod_{j < i}(1-\alpha rel(j|a))
$$

$$
    \alpha \text{-}NDCG@k = \frac{\alpha \text{-}DCG@k} {\alpha \text{-}IDCG@k}
$$

- $a \in A$ are aspects of items i.e. features or subprofiles
- $rel(i|a)$ equals 1 if item at position $i$ is relevant and has aspect $a$ and 0 otherwise
- $\alpha \text{-}IDCG@k = \max(\alpha \text{-}DCG@k)$




## Other

### Coverage

Shows what percentage of items from log appears in recommendations.

`#!py def coverage(items, recs, k=None, user_col='user_id', item_col='item_id'):`

- `items`:
    list of unique item ids.
    
    If recommendations contain new items, not from `items`, metric won't be correct.
    
- `recs`:
    standard user-items dictionary or DataFrame
    
- `k`:
    pass specific k to limit the amount of visible recommendations for each user.

    
### Popularity

Shows mean popularity of recommendations.
 
Scores for items are averaged per recommendation list and globally.

`#!py def popularity(log, pred, k=10, user_col='user_id', item_col='item_id'):`

- `log`: 
    pandas DataFrame with interactions
    
- `pred`: 
    dict of recommendations or DataFrame
    
- `k`: 
    top k items to use from recs

- `user_col`: 
    column name for user ids

- `item_col`:
    column name for item ids

### Surprisal

Measures unexpectedness of recommendations. 

Let $p_k$ be popularity of item $k, p_k \in [0, 1]$. 
Then we can introduce self-information for item $k$ as

$$
    I_k = -log_2(p_k)
$$

We calculate mean self-information for items in recommendation for each user and average it for all users.

In a way it is opposite to the popularity metric.

`#!py def surprisal(log, pred, k=10, user_col='user_id', item_col='item_id'):`

- `log`: 
    pandas DataFrame with interactions
    
- `pred`: 
    dict of recommendations or DataFrame
    
- `k`: 
    top k items to use from recs

- `user_col`: 
    column name for user ids

- `item_col`:
    column name for item ids