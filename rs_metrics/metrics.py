import numpy as np
import pandas as pd

from rs_metrics.helpers import flatten_list, pandas_to_dict, convert_pandas
from rs_metrics.parallel import user_mean, top_k, user_apply, user_mean_sub
from rs_metrics.statistics import item_pop


@convert_pandas
def ndcg(true, pred, k=10):
    """Measures ranking quality"""
    return user_mean(_ndcg_score, true, pred, k)

def _ndcg_score(true, pred):
    k = min(len(true), len(pred))
    gain = pd.Series(pred, dtype=object).isin(true).astype(int)
    discounts = np.log2(np.arange(len(gain)) + 2)
    dcg = np.sum(gain / discounts)
    idcg = np.sum(np.ones(k) / np.log2(np.arange(k) + 2))
    return dcg / idcg

@convert_pandas
def hitrate(true, pred, k=10):
    """Shows what percentage of users has at least one relevant recommendation in their list."""
    return user_mean(_hitrate, true, pred, k)


def _hitrate(true, pred):
    pred = pd.Series(pred, dtype=object)
    true = np.array(true)
    return int(pred.isin(true).any())


@convert_pandas
def precision(true, pred, k=10):
    """Shows what percentage of items in recommendations are relevant, on average."""
    return user_mean(_precision, true, pred, k)


def _precision(true, pred):
    pred = pd.Series(pred, dtype=object)
    return pred.isin(true).mean()


@convert_pandas
def recall(true, pred, k=10):
    """Shows what percentage of relevant items appeared in recommendations, on average."""
    return user_mean(_recall, true, pred, k)


def _recall(true, pred):
    true = pd.Series(true, dtype=object)
    return true.isin(pred).mean()


@convert_pandas
def mrr(true, pred, k=10):
    """Shows inverted position of the first relevant item, on average."""
    return user_mean(_mrr, true, pred, k)


def _mrr(true, pred):
    pred = pd.Series(pred, dtype=object)
    entries = pred.isin(true)
    if entries.any():
        return 1 / (entries.argmax() + 1)
    else:
        return 0


@convert_pandas
def mapr(true, pred, k=10):
    return user_mean(_map, true, pred, k)


def _map(true, pred):
    true = pd.Series(true, dtype=object)
    pred = pd.Series(pred, dtype=object)
    rel = pred.isin(true)
    return (rel.cumsum() / np.arange(1, len(pred) + 1) * rel).mean()


@convert_pandas
def mar(true, pred, k=10):
    return user_mean(_mar, true, pred, k)


def _mar(true, pred):
    true = pd.Series(true, dtype=object)
    pred = pd.Series(pred, dtype=object)
    rel = pred.isin(true)
    return (rel.cumsum() / len(true) * rel).mean()


def coverage(items, recs, k=None, user_col='user_id', item_col='item_id'):
    """What percentage of items appears in recommendations?

    Args:
        items: list of unique item ids
        recs: dict of recommendations
        k: topk items to use from recs

    Returns: float
    """
    if type(recs) is pd.DataFrame:
        recs = pandas_to_dict(recs, user_col, item_col)
    topk = set(flatten_list(top_k(recs, k).values()))
    return pd.Series(items, dtype=object).isin(topk).mean()


def _popularity(df, pred, fill):
    return np.mean([df.get(item, fill) for item in pred])


def popularity(log, pred, k=10, user_col='user_id', item_col='item_id'):
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
    return user_apply(_popularity, scores, pred, k, 0)


def surprisal(log, pred, k=10, user_col='user_id', item_col='item_id'):
    if type(pred) is pd.DataFrame:
        pred = pandas_to_dict(pred, user_col, item_col)
    scores = -np.log2(item_pop(log, user_col, item_col))
    fill = np.log2(log[user_col].nunique())
    return user_apply(_popularity, scores, pred, k, fill)


def a_ndcg(true, pred, aspects, k=10, alpha=0.5, user_col='user_id', item_col='item_id'):
    """Measures redundancy-aware quality and diversity."""
    if type(true) is pd.DataFrame:
        true = pandas_to_dict(true, user_col, item_col)
    if type(pred) is pd.DataFrame:
        pred = pandas_to_dict(pred, user_col, item_col)
    return user_mean_sub(_a_ndcg, true, pred, aspects, k, alpha)


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
