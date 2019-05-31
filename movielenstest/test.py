import nr_lsh as nr
import numpy as np
import time
import pandas
import matplotlib.pyplot as plt

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import perf_eval as pe

num_movies = 10681
num_reviewers = 71567
factors = 50

def main():

    #u, vt, idx_to_id, id_to_movie, review_matrix_csr = load_movielens_files(factors)
    #n = create_tables(vt, num_tables=4, num_partitions=1, bits=64, dim=factors)
    #print(vt.shape)
    #dots = u.dot(vt)
    #real_topk = find_real_topk(dots, k=5)
    #print(dots[0][real_topk[0]])

    #
    # AVERAGE recall across all users.
    # not recall across whole dataset.
    # They should be different, but basically show the same info.
    #

    train, test = load_split_set(factors, 'ratings.dat')
    u, vt, review_matrix_csr = df_to_matrix(train)

    #set2 = load_split_set(50, 'r1.test')
    #_, vt, review_matrix_csr = df_to_matrix(set2)

    test = [tuple(i) for i in test[['userid', 'movieidx', 'rating']].values]

    rec = []
    limit = 21
    for N in range(1, limit):
        rec.append(pe.recall(1000, N, test,
                  item_factors=vt.transpose(),
                  review_matrix_csr=review_matrix_csr))
        print(rec)
    plt.plot(range(1, limit), rec)
    plt.xlabel('N')
    plt.ylabel('Recall(N)')
    plt.show()

    '''
    system_rec = 0
    num_users = 80
    for user in range(num_users):
        user_ratings = review_matrix_csr[user].toarray()[0]
        five_star_indices = pe.find_five_rating(user_ratings)
        user_factors = u[user]
        system_rec += pe.user_recall(1000, 10, user_ratings, user_factors,
                                         item_factors=vt)
    print(system_rec / num_users)
    '''
    #do_other(u, n, idx_to_id, id_to_movie)

def load_split_set(factors, file_name, path='ml-10m/ml-10M100K/'):
    ratings = pandas.read_csv(path + file_name,
                              sep='::',
                              engine='python',
                              names=['userid', 'movieid', 'rating', 'time'])

    movies = pandas.read_csv(path + 'movies.dat',
                             sep='::',
                             engine='python',
                             names=['movieid', 'title', 'genres'])
    movies.insert(0, 'movieidx', range(0, len(movies)))

    df = ratings.join(movies.set_index('movieid'), on='movieid')

    #train = df.sample(frac=0.986, random_state=200)
    train = df.sample(frac=0.986, random_state=200)
    probe = df.drop(train.index) # random subsamle of 1.4% of the dataset.
    test = probe[probe['rating']==5] # test set is all 5 star ratings in probe set.
    return train, test

def df_to_matrix(df):
    #
    # fill sparse matrix with ratings from a dataframe.
    # matr[userid][movieid] = rating.
    #
    ratings = df['rating'].tolist()
    # subtract 1 to map id to index
    users = [id-1 for id in df['userid'].tolist()]
    movies = [id for id in df['movieidx'].tolist()]

    review_matrix_csr = csr_matrix((ratings, (users, movies)),
                                   shape=(num_reviewers, num_movies+1))
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
    num_buckets = int(12000 / num_partitions)
    n = nr.multiprobe(num_tables,
                      num_partitions,
                      bits=bits,
                      dim=dim,
                      num_buckets=num_buckets)
    n.fill(vt.transpose(), False)
    n.stats()
    return n


def find_real_topk(dots, k=1):
    # finds the topk values in each row of dots.
    real_topk = []
    for user in range(dots.shape[0]):
        real_topk.append(dots[user].argsort()[-k:])
    return real_topk


def do_other(us, n, idx_to_id, id_to_movie):
    num_comps = 0
    num_bucks = 0
    num_parts = 0
    num_tables = 0
    successful_count = 0
    for i in range(num_reviewers):
        print(i)

        if (i + 1) % 1000 == 0:
            break

        # make query unit length
        user = us[i] / np.linalg.norm(us[i])

        end = time.time()
        p, stat_tracker = n.k_probe_approx(5, user, 1, 1000)
        end = time.time() - end
        #print(end)

        print('comparisons: ' + str(stat_tracker.get_stats()))

        if p is not None:
            tracked = stat_tracker.tracked_stats()
            num_comps += tracked.comps
            num_bucks += tracked.bucks
            num_parts += tracked.parts
            num_tables += tracked.tables
            successful_count += 1
            for v, idx in p:
                id = idx_to_id[idx]
                movie = id_to_movie[id]
                print(p is not None, movie)

    print(' * Average Comps: ' + str(num_comps / successful_count))
    print(' * Average Puckets Probed: ' + str(num_bucks / successful_count))
    print(' * Average Parts Probed: ' + str(num_parts / successful_count))
    print(' * Average Tables Probed: ' + str(num_tables / successful_count))
    '''
    success, (q, index) = n.probe_approx(user, .005)

    print('Success' if success else 'Fail')

    movies_file = open('ml-10m/ml-10M100K/movies.dat', 'r')
    movies_line = movies_file.readlines()
    movies = [line.split('::') for line in movies_line]

    print(movies[index])
    print(q.dot(user))
    print(vt.transpose()[index].dot(user))
    print(q.dot(vt.transpose()[0]))
    '''


if __name__ == "__main__":
    main()
