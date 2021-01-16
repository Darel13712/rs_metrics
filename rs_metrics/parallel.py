from multiprocessing import Pool, cpu_count
import numpy as np


def user_mean(func, true, pred, k):
    with Pool(cpu_count()) as p:
        return np.mean(
            p.starmap(func, [(true[user], pred.get(user, list())[:k], k) for user in true])
        )



def user_mean_sub(func, true, pred, subprofiles, k, aplha):
    pred = top_k(pred, k)
    with Pool(cpu_count()) as p:
        return np.mean(
            p.starmap(
                func,
                [
                    (true[user], pred.get(user, list()), subprofiles[user], aplha)
                    for user in true
                ],
            )
        )


def user_apply(func, df, pred, k, fill):
    pred = top_k(pred, k)
    with Pool(cpu_count()) as p:
        return np.mean(
            p.starmap(func, [(df, data, fill) for user, data in pred.items()])
        )


def top_k(pred, k):
    with Pool(cpu_count()) as p:
        top_items = p.starmap(get_k, [(data, k) for user, data in pred.items()])
        return {user: top for user, top in zip(pred, top_items)}


def get_k(l, k):
    return l[:k]
