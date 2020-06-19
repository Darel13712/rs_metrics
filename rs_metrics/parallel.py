from multiprocessing import Pool, cpu_count
import numpy as np

from rs_metrics.helpers import flatten_list


def user_mean(func, true, pred, k):
    y = {
        user:
            {
                'true': true[user],
                'pred': pred.get(user, list())[:k]
            }
        for user in true
    }
    with Pool(cpu_count()) as p:
        return np.mean(p.map(func, [data for user, data in y.items()]))


def top_items(pred, k):
    with Pool(cpu_count()) as p:
        topk = p.starmap(get_k, [(data, k) for user, data in pred.items()])
        return flatten_list(topk)


def get_k(l, k):
    return l[:k]