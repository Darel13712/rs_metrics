from multiprocessing import Pool
import numpy as np


def user_parallel(func, true, pred, k, num_workers):
    if num_workers == 1:
        return [func(true[user], pred.get(user, list())[:k], k) for user in true]
    else:
        with Pool(num_workers) as p:
            return p.starmap(func, [(true[user], pred.get(user, list())[:k], k) for user in true])



def user_mean_sub(func, true, pred, subprofiles, k, aplha, num_workers):
    pred = top_k(pred, k, num_workers)
    if num_workers == 1:
        return np.mean([func(true[user], pred.get(user, list()), subprofiles[user], aplha)
                       for user in true])
    else:
        with Pool(num_workers) as p:
            return np.mean(
                p.starmap(
                    func,
                    [
                        (true[user], pred.get(user, list()), subprofiles[user], aplha)
                        for user in true
                    ],
                )
            )


def user_apply(func, df, pred, k, fill, num_workers):
    pred = top_k(pred, k, num_workers)
    if num_workers == 1:
        return np.mean([func(df, data, fill) for user, data in pred.items()])
    else:
        with Pool(num_workers) as p:
            return np.mean(
                p.starmap(func, [(df, data, fill) for user, data in pred.items()])
            )


def top_k(pred, k, num_workers):
    if num_workers == 1:
        return {user: data[:k] for user, data in pred.items()}
    else:
        with Pool(num_workers) as p:
            top_items = p.starmap(get_k, [(data, k) for user, data in pred.items()])
            return {user: top for user, top in zip(pred, top_items)}


def get_k(l, k):
    return l[:k]
