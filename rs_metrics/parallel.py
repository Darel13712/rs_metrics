from multiprocessing import Pool, cpu_count
import numpy as np


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


def unique_items(pred, k):
    with Pool(cpu_count()) as p:
        topk = p.starmap(get_k, [(data, k) for user, data in pred.items()])
        topk = [item for sublist in topk for item in sublist]
        return np.unique(topk)


def get_k(l, k):
    return l[:k]