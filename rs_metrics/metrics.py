import numpy as np
import pandas as pd

from rs_metrics.helpers import flatten_list
from rs_metrics.parallel import user_mean, top_k, user_apply, user_mean_sub
from rs_metrics.statistics import item_pop


def _ndcg_score(data):
    k = min(len(data['pred']), len(data['true']))
    gain = pd.Series(data['pred'], dtype=object).isin(data['true']).astype(int)
    discounts = np.log2(np.arange(len(gain)) + 2)
    dcg = np.sum(gain / discounts)
    idcg = np.sum(np.ones(k) / np.log2(np.arange(k) + 2))
    return dcg / idcg


def ndcg(true, pred, k=10):
    """Measures ranking quality"""
    return user_mean(_ndcg_score, true, pred, k)


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


def a_ndcg(true, pred, aspects, k=10, alpha=0.5):
    """Measures redundancy-aware quality and diversity."""
    return user_mean_sub(_a_ndcg, true, pred, aspects, k, alpha)


def _hitrate(data):
    pred = pd.Series(data['pred'], dtype=object)
    true = np.array(data['true'])
    return int(pred.isin(true).any())


def hitrate(true, pred, k=10):
    """Shows what percentage of users has at least one relevant recommendation in their list."""
    return user_mean(_hitrate, true, pred, k)


def precision(true, pred, k=10):
    """Shows what percentage of items in recommendations are relevant, on average."""
    return user_mean(_precision, true, pred, k)


def _precision(data):
    pred = pd.Series(data['pred'], dtype=object)
    return pred.isin(data['true']).mean()


def recall(true, pred, k=10):
    """Shows what percentage of relevant items appeared in recommendations, on average."""
    return user_mean(_recall, true, pred, k)


def _recall(data):
    true = pd.Series(data['true'], dtype=object)
    return true.isin(data['pred']).mean()


def _mrr(data):
    pred = pd.Series(data['pred'], dtype=object)
    entries = pred.isin(data['true'])
    if entries.any():
        return 1 / (entries.argmax() + 1)
    else:
        return 0


def mrr(true, pred, k=10):
    """Shows inverted position of the first relevant item, on average."""
    return user_mean(_mrr, true, pred, k)


def _map(data):
    true = pd.Series(data['true'], dtype=object)
    pred = pd.Series(data['pred'], dtype=object)
    rel = pred.isin(true)
    return (rel.cumsum() / np.arange(1, len(pred) + 1) * rel).mean()


def mapr(true, pred, k=10):
    return user_mean(_map, true, pred, k)


def _mar(data):
    true = pd.Series(data['true'], dtype=object)
    pred = pd.Series(data['pred'], dtype=object)
    rel = pred.isin(true)
    return (rel.cumsum() / len(true) * rel).mean()


def mar(true, pred, k=10):
    return user_mean(_mar, true, pred, k)


def coverage(items, recs, k=None):
    """What percentage of items appears in recommendations?

    Args:
        items: list of unique item ids
        recs: dict of recommendations
        k: topk items to use from recs

    Returns: float
    """
    topk = set(flatten_list(top_k(recs, k).values()))
    return pd.Series(items, dtype=object).isin(topk).mean()


def _popularity(df, pred, fill):
    return np.mean([df.get(item, fill) for item in pred])


def popularity(log, pred, k=10, user_col='user_id', item_col='item_id'):
    """
    Mean popularity of recommendations.

    Args:
        log: pandas DataFrame with interactions
        pred: dict of recommendations
        k: top k items to use from recs
        user_col: column name for user ids
        item_col: column name for item ids

    """
    scores = item_pop(log, user_col, item_col)
    return user_apply(_popularity, scores, pred, k, 0)


def surprisal(log, pred, k=10, user_col='user_id', item_col='item_id'):
    scores = -np.log2(item_pop(log, user_col, item_col))
    fill = np.log2(log[user_col].nunique())
    return user_apply(_popularity, scores, pred, k, fill)
