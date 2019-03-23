

import nr_lsh as nr
import numpy as np
import time

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def main():
    num_movies = 65133
    num_reviewers = 71567 

    ratings_file = open('ml-10m/ml-10M100K/ratings.dat', 'r')

    ratings_line = ratings_file.readlines()

    ratings = [line.split('::') for line in ratings_line]
    ratings = [tuple(map(float, r[:-1])) for r in ratings]

    data = [r[2] for r in ratings]
    user_id = [int(r[0])-1 for r in ratings]
    movie_id = [int(r[1])-1 for r in ratings]

    review_matrix_csr = csr_matrix((data, (user_id, movie_id)), 
                                   shape=(num_reviewers, num_movies))

    u, s, vt = svds(review_matrix_csr, k=200)

    n = nr.multiprobe(256, 6, 200, 75)

    n.fill(vt.transpose(), False)

    n.stats()

    num_comps = 0
    for i in range(num_reviewers):
        print(i)

        if (i+1) % 100 == 0:
            break

        user = u[i]

        end = time.time()
        succes, p, stat_tracker = n.k_probe_approx(1, user, .005)
        end = time.time() - end
        #print(end)

        #print('comparisons: ' + str(stat_tracker.get_stats()))

        comps, = stat_tracker.get_stats()

        num_comps += comps

        #movies_file = open('ml-10m/ml-10M100K/movies.dat', 'r')
        #movies_line = movies_file.readlines()
        #movies = [line.split('::') for line in movies_line]
        #for v, i in p:
        #    print(movies[i])

        #end = time.time()
        #true_max = n.find_max_inner(user)
        #end = time.time() - end
        #print(end)
        #print(movies[true_max[1]])
        #print(user.dot(true_max[0]))

    print(num_comps / 100)

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
