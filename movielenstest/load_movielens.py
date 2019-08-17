import nr_lsh as nr
import numpy as np
import time
import pandas

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

num_movies = 10681
num_reviewers = 71567


def load(num_tables, num_partitions, bits, factors):
    train, test, mean_rating = load_split_set(factors,
                                              file_name='ratings.dat',
                                              frac=0.986)
    u, vt, review_matrix_csr = df_to_matrix(train, factors)
    test = [tuple(i) for i in test[['userid', 'movieidx', 'rating']].values]
    # returns the test set, "trained" table, and vt.
    return test, create_tables(vt, num_tables, num_partitions, bits,
                               factors), vt


def load_split_set(factors, file_name, frac=0.986, path='ml-10m/ml-10M100K/'):
    ratings = pandas.read_csv(path + file_name,
                              sep='::',
                              engine='python',
                              names=['userid', 'movieid', 'rating', 'time'])
    ratings['userid'] = ratings['userid'].subtract(1)  # map to index

    # center ratings around the overall mean.
    mean_rating = ratings['rating'].mean()
    ratings['rating'] = ratings['rating'].subtract(mean_rating)

    movies = pandas.read_csv(path + 'movies.dat',
                             sep='::',
                             engine='python',
                             names=['movieid', 'title', 'genres'])
    # add column that tracks index in the movies.dat file.
    movies.insert(0, 'movieidx', range(0, len(movies)))

    df = ratings.join(movies.set_index('movieid'), on='movieid')

    train = df.sample(frac=frac, random_state=200)
    probe = df.drop(train.index)  # random subsamle of 1.4% of the dataset.

    # test set is all 5 (minus the mean) star ratings in probe set.
    test = probe[probe['rating'] == 5 - mean_rating]

    return train, test, mean_rating


def df_to_matrix(df, factors):
    #
    # fill sparse matrix with ratings from a dataframe.
    # matr[userid][movieid] = rating.
    #
    ratings = df['rating'].tolist()
    # subtract 1 to map id to index
    users = [id for id in df['userid'].tolist()]
    movies = [id for id in df['movieidx'].tolist()]

    review_matrix_csr = csr_matrix((ratings, (users, movies)),
                                   shape=(num_reviewers, num_movies + 1))
    u, s, vt = svds(review_matrix_csr, k=factors)
    user_factors = csr_matrix.dot(u, s)
    item_factors = vt
    return user_factors, item_factors, review_matrix_csr


def create_tables(vt, num_tables, num_partitions, bits, dim):
    '''
    simple function to create the tables.
    '''
    # more partitions = less items per partition, so number of buckets
    # should be fewer.
    num_buckets = int(10681 / num_partitions)
    n = nr.multiprobe(num_tables,
                      num_partitions,
                      bits=bits,
                      dim=dim,
                      num_buckets=2 * num_buckets)
    n.fill(vt.transpose(), False)
    n.stats()
    return n
