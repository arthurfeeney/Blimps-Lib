import nr_lsh as nr
import numpy as np
import time
import pandas

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import perf_eval as pe

num_movies = 10681
num_reviewers = 71567
factors = 50

def main():

    u, vt, idx_to_id, id_to_movie, review_matrix_csr = load_movielens_files(factors)
    #n = create_tables(vt, num_tables=4, num_partitions=1, bits=64, dim=factors)
    #print(vt.shape)
    #dots = u.dot(vt)
    #real_topk = find_real_topk(dots, k=5)
    #print(dots[0][real_topk[0]])

    user_ratings = review_matrix_csr[0].toarray()[0]

    five_star_indices = pe.find_five_rating(user_ratings)

    user_factors = u[0]


    #
    # AVERAGE recall of users.
    # not recall across whole dataset.
    # I believe they are distinct.
    #

    system_rec = 0
    num_users = 40
    for user in range(num_users):
        user_ratings = review_matrix_csr[user].toarray()[0]
        five_star_indices = pe.find_five_rating(user_ratings)
        user_factors = u[user]
        system_rec += pe.user_recall(1000, 1, user_ratings, user_factors,
                                         item_factors=vt)
    print(system_rec / num_users)
    #do_other(u, n, idx_to_id, id_to_movie)


def load_movielens_files(factors):
    '''
    loads the mocites lens rating and movies files.
    computes the SVD of user-review matrix. user-review = u*s*v.T
    Returns u*s and s.T
    '''
    ratings_file = open('ml-10m/ml-10M100K/ratings.dat', 'r')
    ratings_line = ratings_file.readlines()
    ratings = [line.split('::') for line in ratings_line]
    ratings = [tuple(map(float, r[:-1])) for r in ratings]

    data = [r[2] for r in ratings]
    # users and moves are 1-indexed in the file, so subtract 1.
    user_id = [int(r[0]) - 1 for r in ratings]
    movie_id = np.array([int(r[1]) - 1 for r in ratings])

    movies_file = open('ml-10m/ml-10M100K/movies.dat', 'r')
    movies_line = movies_file.readlines()
    movies = [line.split('::') for line in movies_line]

    movie_id_list = [int(m[0]) - 1 for m in movies]
    movie_indices = range(num_movies)

    id_to_idx = dict(zip(movie_id_list, movie_indices))
    idx_to_id = dict(zip(movie_indices, movie_id_list))

    # make the movie ids their indices in the file.
    movie_id = [id_to_idx[m] for m in movie_id]

    # map an id to a movie
    id_to_movie = dict([(int(m[0]) - 1, [m[1], m[2]]) for m in movies])

    # large sparse matrix.
    review_matrix_csr = csr_matrix((data, (user_id, movie_id)),
                                   shape=(num_reviewers, num_movies))

    u, s, vt = svds(review_matrix_csr, k=factors)

    u = csr_matrix.dot(u, s)

    return u, vt, idx_to_id, id_to_movie, review_matrix_csr


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
