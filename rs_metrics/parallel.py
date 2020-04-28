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
