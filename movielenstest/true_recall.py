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

#num_movies = 10681
#num_reviewers = 71567
factors = 50


def main():
    train, test, mean_rating = load_split_set(factors,
                                              file_name='ratings.dat',
                                              frac=0.986)
    u, vt, review_matrix_csr = df_to_matrix(train, factors)
    test = [tuple(i) for i in test[['userid', 'movieidx', 'rating']].values]

    n = create_tables(vt, 20, 1, 32, 50)

    rec = pe.recall(1000, range(1, 6), test,
                    item_factors=vt.transpose(),
                    review_matrix_csr=review_matrix_csr,
                    mean_rating=mean_rating)
    prec = pe.precision(1000, range(1, 6), test,
                        item_factors=vt.transpose(),
                        review_matrix_csr=review_matrix_csr,
                        mean_rating=mean_rating)

    #
    # Display Recall at N
    #

    plt.plot(list(rec.keys()), list(rec.values()))
    plt.xlabel('N')
    plt.ylabel('Recall(N)')
    plt.xticks([0, 5, 10, 15, 20])
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.show()

    #
    # Display Precision vs Recall
    # Recall should approach 1.0 for this plot.
    #

    plt.plot(list(rec.values()), list(prec.values()))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xticks(np.arange(0, 1.1, step=0.2))
    plt.show()

if __name__ == "__main__":
    main()
