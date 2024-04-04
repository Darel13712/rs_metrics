import numpy as np
import pandas as pd
import pytest

from rs_metrics.metrics import _ndcg_score
from rs_metrics import *
from rs_metrics.statistics import item_pop


def test_dcg_score_1():
    assert _ndcg_score([1], [1], 1) == 1


def test_dcg_score_0():
    assert _ndcg_score([1], [0], 1) == 0


def test_dcg_score_half():
    idcg2 = (1 / np.log2(2) + 1 / np.log2(3))
    dcg = 1 / np.log2(3)
    assert _ndcg_score([1, 2], [0, 2], 2) == dcg / idcg2


def test_ndcg_test_less_than_k():
    y_true = {1: [1, 2, 3]}
    assert ndcg(y_true, y_true, 5) == ndcg(y_true, y_true, 3) == 1


def test_ndcg():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [1, 2], 2: [0, 0]}
    assert ndcg(y_true, y_pred, 2) == 0.5


def test_ndcg_pandas():
    y_true = pd.DataFrame([[1, 1], [1, 2]], columns=['user_idx', 'item_id'])
    y_pred = pd.DataFrame([[1, 1], [1, 0]], columns=['user_idx', 'item_id'])
    idcg2 = (1 / np.log2(2) + 1 / np.log2(3))
    dcg = 1 / np.log2(2)
    assert ndcg(y_true, y_pred, 2, user_col='user_idx') == dcg / idcg2


def test_a_ndcg_one_user():
    y_true = {1: [1, 2, 3]}
    y_pred = {1: [1, 2, 3]}
    sp = {1: [{1}, {2}, {3}]}
    assert a_ndcg(y_true, y_pred, sp, 3) == 1


def test_a_ndcg():
    y_true = {1: [1, 2, 3], 2: [1, 2, 3]}
    y_pred = {1: [1, 2, 3], 2: [0, 0, 0]}
    sp = {1: [{1, 2}, {3}], 2: [{1, 2, 3}]}
    u1_score = (1 + 0.4/np.log2(3) + 1/np.log2(4)) / (1 + 1/np.log2(3) + 0.4/np.log2(4))
    answer = (u1_score + 0) / 2
    assert a_ndcg(y_true, y_pred, sp, 3, 0.6) == answer


def test_hitrate():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [0, 1], 2: [0, 0]}
    assert hitrate(y_true, y_pred, 2) == 0.5

def test_apply_mean():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [0, 1], 2: [0, 0]}
    res = hitrate(y_true, y_pred, 2, apply_mean=False)
    assert len(res) == 2
    res_averaged = hitrate(y_true, y_pred, 2)
    assert np.mean(res) == res_averaged


def test_precision():
    y_true = {1: [1, 0, 0, 2], 2: [1, 2]}
    y_pred = {1: [1, 2], 2: [1, 3]}
    assert precision(y_true, y_pred, 2) == 0.75


def test_recall():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [1, 3], 2: [0, 0]}
    assert recall(y_true, y_pred, 2) == 0.25


def test_mrr():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [1, 3], 2: [0, 1]}
    assert mrr(y_true, y_pred, 2) == 0.75


def test_map():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [1, 3], 2: [0, 1]}
    assert mapr(y_true, y_pred, 2) == 0.375


def test_coverage():
    items = [1, 2, 3, 4]
    pred = {1: [1, 2], 2: [2, 5]}
    assert coverage(items, pred) == 0.5


@pytest.fixture
def log():
    return pd.DataFrame({'user_id': [1, 1, 2], 'item_id': [1, 2, 2]})


def test_item_pop(log):
    pops = item_pop(log)
    assert sum(pops) == 1.5


def test_popularity(log):
    pred = {1: [2], 2: [1]}
    assert popularity(log, pred, 2) == 0.75


def test_surprisal():
    df = pd.DataFrame({'user_id': [1, 2], 'item_id': [1, 2]})
    pred = {1: [2], 2: [1]}
    assert surprisal(df, pred, 2) == 1
