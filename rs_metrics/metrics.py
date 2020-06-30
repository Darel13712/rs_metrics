import numpy as np
import pandas as pd

from rs_metrics.helpers import flatten_list
from rs_metrics.parallel import user_mean, top_k, user_apply
from rs_metrics.statistics import item_pop


def _dcg_score(data):
    y = pd.Series(data['pred']).isin(data['true']).astype(int)
    gain = 2 ** y - 1
    discounts = np.log2(np.arange(len(y)) + 2)
    return np.sum(gain / discounts)


def ndcg(true, pred, k=10):
    """Measures ranking quality"""
    score = user_mean(_dcg_score, true, pred, k)
    idcg = np.sum(np.ones(k) / np.log2(np.arange(k) + 2))
    return score / idcg


def _hitrate(data):
    pred = pd.Series(data['pred'])
    true = np.array(data['true'])
    return int(pred.isin(true).any())


def hitrate(true, pred, k=10):
    """Shows what percentage of users has at least one relevant recommendation in their list."""
    return user_mean(_hitrate, true, pred, k)


def precision(true, pred, k=10):
    """Shows what percentage of items in recommendations are relevant, on average."""
    return user_mean(_precision, true, pred, k)


def _precision(data):
    pred = pd.Series(data['pred'])
    return pred.isin(data['true']).mean()


def recall(true, pred, k=10):
    """Shows what percentage of relevant items appeared in recommendations, on average."""
    return user_mean(_recall, true, pred, k)


def _recall(data):
    true = pd.Series(data['true'])
    return true.isin(data['pred']).mean()


def _mrr(data):
    pred = pd.Series(data['pred'])
    entries = pred.isin(data['true'])
    if entries.any():
        return 1 / (entries.argmax() + 1)
    else:
        return 0


def mrr(true, pred, k=10):
    """Shows inverted position of the first relevant item, on average."""
    return user_mean(_mrr, true, pred, k)


def _map(data):
    true = pd.Series(data['true'])
    pred = pd.Series(data['pred'])
    rel = pred.isin(true)
    return (rel.cumsum() / np.arange(1, len(pred) + 1) * rel).mean()


def mapr(true, pred, k=10):
    return user_mean(_map, true, pred, k)


def _mar(data):
    true = pd.Series(data['true'])
    pred = pd.Series(data['pred'])
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
    return pd.Series(items).isin(topk).mean()


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