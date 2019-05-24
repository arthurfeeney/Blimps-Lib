
import sys
sys.path.append('../movielenstest')

import pytest
import numpy as np
import perf_eval as pe

#class TestPerfEval(object):
def test_simple_find_five_rating():
    matr = np.array([[1,2,3,5],[5,0,0,1]])
    indices = pe.find_five_rating(0, matr)
    assert indices == [3]

    indices = pe.find_five_rating(1, matr)
    assert indices == [0]


def test_get_k_unrated():
    # for my application, the mmatrix is super sparse. Each row is
    # guaranteed to have a 0.
    matr = np.array([[1, 0, 5, 0], [3, 0, 3, 0]])
    unrated_indices = pe.get_k_unrated(k=2, user=0, rating_matrix=matr)
    assert all(unrated_indices == [1, 3])

    unrated_indices = pe.get_k_unrated(k=2, user=1, rating_matrix=matr)
    assert all(unrated_indices == [1, 3])
