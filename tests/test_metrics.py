import numpy as np
import pytest

from rs_metrics.metrics import _dcg_score
from rs_metrics import *


@pytest.fixture
def inner_dict():
    def func(pred, true):
        return {'pred': pred, 'true': true}

    return func


def test_dcg_score_1(inner_dict):
    assert _dcg_score(inner_dict([1], [1])) == 1


def test_dcg_score_0(inner_dict):
    assert _dcg_score(inner_dict([1], [0])) == 0


def test_dcg_score_half(inner_dict):
    assert _dcg_score(inner_dict([1, 2], [0, 2])) == 1 / np.log2(3)


def test_ndcg():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [1, 2], 2: [0, 0]}
    assert ndcg(y_true, y_pred, 2) == 0.5


def test_hitrate():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [0, 1], 2: [0, 0]}
    assert hitrate(y_true, y_pred, 2) == 0.5


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


def test_mar():
    y_true = {1: [1, 2], 2: [1, 2]}
    y_pred = {1: [1, 3], 2: [0, 1]}
    assert mar(y_true, y_pred, 2) == 0.25
