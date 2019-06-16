import nr_lsh as nr
import numpy as np
import time
import pandas
import matplotlib.pyplot as plt

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from load_movielens import *
import perf_eval as pe



def main():
    factors = 50


    train, test, mean_rating = load_split_set(factors,
                                              file_name='ratings.dat',
                                              frac=0.986)
    u, vt, review_matrix_csr = df_to_matrix(train, factors)
    test = [tuple(i) for i in test[['userid', 'movieidx', 'rating']].values]

    n = create_tables(vt, 20, 1, 32, 50)

    mips_recall, hit_recall = pe.MIPS_recall(20, test,
                                             item_factors=vt.transpose(),
                                             nr_table=n,
                                             review_matrix_csr=review_matrix_csr,
                                             mean_rating=mean_rating)
    print(mips_recall)
    print(hit_recall)


if __name__ == "__main__":
    main()
