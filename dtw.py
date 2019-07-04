import numpy as np
from timewarp.timewarp import create_dist_mat


def is_out(i, j, shape, r=14):
    """Have we crossed outside the window?
    """
    return abs(shape[1] * i / shape[0] - j) > r


def not_finished(i, j, shape):
    """True if the walk is still within bounds
    """
    cond = (not is_out(i, j, shape)) and (i < (shape[0] - 1)) and (j < (shape[1] - 1))
    return cond


def step(dist, i, j):
    """
    create all valid moves (N, E, NE),
    return the move corresponding to min of dist
    """
    moves = [
        (mi, mj)
        for mi, mj in ([i + 1, j], [i, j + 1], [i + 1, j + 1])
        if mi < dist.shape[0] and mj < dist.shape[1]
    ]
    vals = [dist[mi, mj] for mi, mj in moves]
    return moves[np.argmin(np.array(vals))]


def walk_dist_mat(dist):
    """
    perform walk across the distance matrix
    """
    i, j = 0, 0
    shp = np.array(dist.shape)
    while not_finished(i, j, shp):
        i, j = step(dist, i, j)
        out = is_out(i, j, shp)

    return dist[i, j] / np.sum(shp) if not out else np.inf


def compare(ref, test):
    """
    compare 2 utterances
    """
    return walk_dist_mat(np.asarray(create_dist_mat(ref, test)))


def find_match(ref_set, query):
    """Find ref utterance with min distance to query
    """
    return min(ref_set.items(), key=lambda r: compare(*r[1], query))[0]
