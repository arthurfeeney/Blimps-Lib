
import sys
sys.path.append('../movielenstest')

import pytest
import numpy as np
import perf_eval as pe

#class TestPerfEval(object):
def test_simple_find_five_rating():
    matr = np.array([[1,2,3,5],[5,0,0,1]])
    indices = pe.find_five_rating(matr[0])
    assert indices == [3]

    indices = pe.find_five_rating(matr[1])
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
    assert five_rating == 0.0
    assert others[0] == user_factors.dot(item_factors[:,3])
    assert others[1] == user_factors.dot(item_factors[:,1])

def test_hit():
    # input list should be in sorted, ascending order for it to work properly. 
    assert pe.hit(3, 5.1, [1, 2, 3, 4, 5, 6])
    assert pe.hit(2, 5.1, [1, 2, 3, 4, 5, 6])
    assert not pe.hit(1, 5.1, [1, 2, 3, 4, 5, 6])
