from functools import lru_cache

import numpy as np
import pandas as pd

from rs_metrics.helpers import flatten_list, pandas_to_dict, convert_pandas, maybe_mean
from rs_metrics.parallel import user_parallel, top_k, user_apply, user_mean_sub
from rs_metrics.statistics import item_pop


@convert_pandas
def ndcg(true, pred, k=10, num_workers=1, apply_mean=True):
    """Measures ranking quality"""
    res = user_parallel(_ndcg_score, true, pred, k, num_workers)
    return maybe_mean(res, apply_mean)

def _ndcg_score(true, pred, k):
    true = set(true)
    gain = [item in true for item in pred]
    dcg = np.sum(gain / _discounts(len(gain)))
    return dcg / _idcg(min(len(true), k))

@lru_cache(maxsize=128)
def _discounts(k):
    return np.log2(np.arange(k) + 2)

@lru_cache(maxsize=128)
def _idcg(k):
    return np.sum(np.ones(k) / np.log2(np.arange(k) + 2))

@convert_pandas
def hitrate(true, pred, k=10, num_workers=1, apply_mean=True):
    """Shows what percentage of users has at least one relevant recommendation in their list."""
    res = user_parallel(_hitrate, true, pred, k, num_workers)
    return maybe_mean(res, apply_mean)


def _hitrate(true, pred, k):
    return int(np.isin(pred, true).any())


@convert_pandas
def precision(true, pred, k=10, num_workers=1, apply_mean=True):
    """Shows what percentage of items in recommendations are relevant, on average."""
    res = user_parallel(_precision, true, pred, k, num_workers)
    return maybe_mean(res, apply_mean)



def _precision(true, pred, k):
    return np.isin(pred, true).mean()


@convert_pandas
def recall(true, pred, k=10, num_workers=1, apply_mean=True):
    """Shows what percentage of relevant items appeared in recommendations, on average."""
    res = user_parallel(_recall, true, pred, k, num_workers)
    return maybe_mean(res, apply_mean)


def _recall(true, pred, k):
    return np.isin(true, pred).mean()


@convert_pandas
def mrr(true, pred, k=10, num_workers=1, apply_mean=True):
    """Shows inverted position of the first relevant item, on average."""
    res = user_parallel(_mrr, true, pred, k, num_workers)
    return maybe_mean(res, apply_mean)


def _mrr(true, pred, k):
    entries = np.isin(pred, true)
    if entries.any():
        return 1 / (entries.argmax() + 1)
    else:
        return 0


@convert_pandas
def mapr(true, pred, k=10, num_workers=1, apply_mean=True):
    res = user_parallel(_map, true, pred, k, num_workers)
    return maybe_mean(res, apply_mean)


def _map(true, pred, k):
    rel = np.isin(pred, true)
    if len(true) > 0:
        return (rel.cumsum() / np.arange(1, len(pred) + 1) * rel).sum() / len(true)
    else:
        return 0.0


def coverage(items, recs, k=None, num_workers=1, user_col='user_id', item_col='item_id'):
    """What percentage of items appears in recommendations?

    Args:
        items: list of unique item ids
        recs: dict of recommendations
        k: topk items to use from recs

    Returns: float
    """
    if type(recs) is pd.DataFrame:
        recs = pandas_to_dict(recs, user_col, item_col)
    topk = list(set(flatten_list(top_k(recs, k, num_workers).values())))
    return np.isin(items, topk).mean()


def _popularity(df, pred, fill):
    return np.mean([df.get(item, fill) for item in pred])


def popularity(log, pred, k=10, num_workers=1, user_col='user_id', item_col='item_id'):
    """
    Mean popularity of recommendations.

    Args:
        log: pandas DataFrame with interactions
        pred: pandas DataFrame with recommendations
        k: top k items to use from recs
        user_col: column name for user ids
        item_col: column name for item ids

    """
    if type(pred) is pd.DataFrame:
        pred = pandas_to_dict(pred, user_col, item_col)
    scores = item_pop(log, user_col, item_col)
    return user_apply(_popularity, scores, pred, k, 0, num_workers)


def surprisal(log, pred, k=10, num_workers=1, user_col='user_id', item_col='item_id'):
    if type(pred) is pd.DataFrame:
        pred = pandas_to_dict(pred, user_col, item_col)
    scores = -np.log2(item_pop(log, user_col, item_col))
    fill = np.log2(log[user_col].nunique())
    return user_apply(_popularity, scores, pred, k, fill, num_workers)


def a_ndcg(true, pred, aspects, k=10, alpha=0.5, num_workers=1, user_col='user_id', item_col='item_id'):
    """Measures redundancy-aware quality and diversity."""
    if type(true) is pd.DataFrame:
        true = pandas_to_dict(true, user_col, item_col)
    if type(pred) is pd.DataFrame:
        pred = pandas_to_dict(pred, user_col, item_col)
    return user_mean_sub(_a_ndcg, true, pred, aspects, k, alpha, num_workers)


def _a_ndcg(true, pred, aspects, alpha):
    p = pd.Series(pred, dtype=object)
    penalty = 1 - alpha
    dcg = 0
    hits = p.isin(true).astype(int)
    num_aspects = len(aspects)
    k = len(pred)

    fit = k // num_aspects
    extra = k % num_aspects
    idcg = np.append(np.ones(fit * num_aspects), [penalty ** fit] * extra)
    idcg /= np.log2(np.arange(k) + 2)
    idcg = idcg.sum()

    for aspect in aspects:
        items_from_aspects = p.isin(aspect)
        if items_from_aspects.any():
            aspect_positions = pd.Series(np.NaN, index=p.index, dtype=float)
            aspect_positions[items_from_aspects] = range(items_from_aspects.sum())
            gain = hits * (penalty ** aspect_positions).fillna(0)
            discounts = np.log2(np.arange(len(gain)) + 2)
            dcg += np.sum(gain / discounts)
    return dcg / idcg
