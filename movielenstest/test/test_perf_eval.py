
import sys
sys.path.append('../movielenstest')

import pytest
import numpy as np
import perf_eval as pe

#class TestPerfEval(object):
def test_simple_find_five_rating():
    matr = np.array([[1,2,3,5],[5,0,0,1]], dtype=np.float64)
    mean_rating = matr[matr != 0].mean()
    matr[matr != 0] -= mean_rating
    indices = pe.find_five_rating(matr[0], mean_rating)
    assert indices == [3]

    indices = pe.find_five_rating(matr[1], mean_rating)
    assert indices == [0]


def test_get_k_unrated():
    # for my application, the mmatrix is super sparse. Each row is
    # guaranteed to have a 0.
    matr = np.array([[1, 0, 5, 0], [3, 0, 3, 0]])
    unrated_indices = pe.get_k_unrated(k=2, user_ratings=matr[0])
    assert all(np.sort(unrated_indices) == [1, 3])

    unrated_indices = pe.get_k_unrated(k=2, user_ratings=matr[0])
    assert all(np.sort(unrated_indices) == [1, 3])

def test_random_item_ranked_list():
    k = 2
    user_ratings = np.array([1,0,5,0])
    five_star_idx = 2
    user_factors = np.array([.3,.3])
    item_factors = np.array([
        [.5, .5],
        [.3, .3],
        [.1, -.1],
        [.2, .2],
    ]).transpose()
    (five_rating, others) = pe.random_item_ranked_list(k, user_ratings, five_star_idx, user_factors, item_factors)
    assert five_rating == user_ratings.dot(item_factors.transpose()).dot(item_factors[:,2])


def test_hit():
    # input list should be in sorted, ascending order for it to work properly.
    assert pe.hit(3, 5.1, [1, 2, 3, 4, 5, 6])
    assert pe.hit(2, 5.1, [1, 2, 3, 4, 5, 6])
    assert not pe.hit(1, 5.1, [1, 2, 3, 4, 5, 6])
    assert not pe.hit(2, 4.1, [1, 2, 3, 4, 5, 6])

def test_user_recall():
    k = 4
    factors = 4
    user_ratings = np.array([
        [0, 0, 0, 0, 0, 5],
        [3, 5, 2, 5, 2, 3],
        [3, 3, 2, 0, 2, 0],
        [5, 0, 0, 0, 2, 1],
        [3, 5, 0, 5, 0, 2]
    ], dtype=np.float64)
    u, s, vt = np.linalg.svd(user_ratings, full_matrices=True)
    u, s, vt = u[:,:factors], s[:factors], vt[:factors]
    user_factors = u * s
    item_factors = vt
    mean_rating = user_ratings[user_ratings != 0].mean()
    user_ratings[user_ratings != 0] -= mean_rating

    assert pe.find_five_rating(user_ratings[0], mean_rating) == [5]
    assert pe.find_five_rating(user_ratings[1], mean_rating) == [1, 3]
    assert pe.find_five_rating(user_ratings[2], mean_rating) == []
    assert pe.find_five_rating(user_ratings[3], mean_rating) == [0]
    assert pe.find_five_rating(user_ratings[4], mean_rating) == [1, 3]

    # multiple trials since get_k_unrated gets a random subset.
    for trial in range(20):
        for elem in pe.get_k_unrated(4, user_ratings[0]):
            assert elem in [0,1,2,3,4]

    assert pe.user_recall(k, 1, user_ratings[0], user_factors[0,:], item_factors, mean_rating) == 1

def test_topk_inner():
    assert all(pe.topk_inner(2, np.array([1,4,3,2]))[0] == np.array([4, 3]))
    assert all(pe.topk_inner(1, np.array([1,4,3,2]))[0] == np.array([4]))
    assert all(pe.topk_inner(4, np.array([1,4,3,2,-3, 5.5, 6.2]))[0] == np.array([6.2, 5.5, 4, 3]))
    assert all(pe.topk_inner(2, np.array([3,1,2,4,0]))[1] == np.array([3,0]))

def test_fraction_intersect():
    assert pe.fraction_intersect([1,2,3],[1,2,3]) == 1
    assert pe.fraction_intersect([1,2,3], [1,2]) == 2/3
    assert pe.fraction_intersect([0,1,2,6,3,7,8], [0,6,7,8]) == 4/7
    assert pe.fraction_intersect([0,1,2,6,3,7,8], [0,6,7,8,5,10]) == 4/7
    assert pe.fraction_intersect([1,2,3,4],[5,6,7,8]) == 0
