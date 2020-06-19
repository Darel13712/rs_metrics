# Metrics

## Relevance 

### HitRate

`#!py def hitrate(true, pred, k=10):`

Shows what percentage of users has at least one relevant recommendation in their list.

$$
    HitRate@k =  \frac {\sum \limits_{u \in U} \max \limits_{i \in L(u)}(rel(i, u))} {|U|}
$$

- $rel(i, u)$ equals 1 if item $i$ is relevant to user $u$.
- $L(u)$ is a recommendation list of length $k$ for user $u$.


### Precision

`#!py def precision(true, pred, k=10):`

Shows what percentage of items in recommendations are relevant, on average.

$$
Precision@k = \frac{1}{|U|} \sum_{u \in U} \frac{|rel_u \cap rec_u|}{|rec_u|}
$$

- $rel_u$ – items relevant for user $u$
- $rec_u$ – top-k items recommended to user $u$

### Mean Average Precision

`#!py def mapr(true, pred, k=10):`

$$
AP@k(u) = \frac{1}{k} \sum_{i \in rec_u} rel(i, u)Precision@pos_{i, u}(u)
$$

$$
MAP@k = \frac{1}{|U|} \sum_{u \in U} AP@k(u)
$$

- $rel(i, u)$ – equals 1 if item $i$ is relevant to user $u$
- $rec_u$ – top-k recommendations for user $u$
- $pos_{i, u}$ – position of item $i$ in recommendation list $rec_u$

### Recall

`#!py def recall(true, pred, k=10):`

Shows what percentage of relevant items appeared in recommendations, on average.

$$
Recall@k = \frac{1}{|U|} \sum_{u \in U} \frac{|rel_u \cap rec_u|}{|rel_u|}
$$

- $rel_u$ – items relevant for user $u$
- $rec_u$ – top-k items recommended to user $u$

### Mean Average Recall

`#!py def mar(true, pred, k=10):`

$$
AR@k(u) = \frac{1}{k} \sum_{i \in rec_u} rel(i, u)Recall@pos_{i, u}(u)
$$

$$
MAR@k = \frac{1}{|U|} \sum_{u \in U} AR@k(u)
$$

- $rel(i, u)$ – equals 1 if item $i$ is relevant to user $u$
- $rec_u$ – top-k recommendations for user $u$
- $pos_{i, u}$ – position of item $i$ in recommendation list $rec_u$

## Ranking
 
### NDCG
`#!py def ndcg(true, pred, k=10):`

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

`#!py def mrr(true, pred, k=10):`

Mean Reciprocal Rank shows inverted position of the first relevant item, on average.

## Other

### Coverage

Shows what percentage of items from log appears in recommendations.

`#!py def coverage(items, recs, k=None):`

- `items`:
    list of unique item ids.
    
    If recommendations contain new items, not from `items`, metric won't be correct.
    
- `recs`:
    standard user-items dictionary
    
- `k`:
    pass specific k to limit the amount of visible recommendations for each user.

    
    