

import nr_lsh as nr
import numpy as np
import time
import pandas

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def main():
    num_movies = 10681
    num_reviewers = 71567 

    ratings_file = open('ml-10m/ml-10M100K/ratings.dat', 'r')

    ratings_line = ratings_file.readlines()

    ratings = [line.split('::') for line in ratings_line]
    ratings = [tuple(map(float, r[:-1])) for r in ratings]

    data = [r[2] for r in ratings]
    # users and moves are 1-indexed, so subtract 1. 
    user_id = [int(r[0])-1 for r in ratings]
    movie_id = np.array([int(r[1])-1 for r in ratings])

    movies_file = open('ml-10m/ml-10M100K/movies.dat', 'r')
    movies_line = movies_file.readlines()
    movies = [line.split('::') for line in movies_line]

    movie_id_list = [int(m[0])-1 for m in movies]
    movie_indices = range(num_movies)


    id_to_idx = dict(zip(movie_id_list, movie_indices))
    idx_to_id = dict(zip(movie_indices, movie_id_list))

    # make the movie ids their indices in the file. 
    movie_id = [id_to_idx[m] for m in movie_id]

    # map an id to a movie
    id_to_movie = dict( [(int(m[0])-1, [m[1], m[2]]) for m in movies])


    review_matrix_csr = csr_matrix((data, (user_id, movie_id)), 
                                   shape=(num_reviewers, num_movies))

    u, s, vt = svds(review_matrix_csr, k=150)
   
    u = csr_matrix.dot(u, s)
    n = nr.multiprobe(10, 100, 150, 10000)

    print(num_reviewers)
    print(num_movies)
    print(vt.shape)

    n.fill(vt.transpose(), False)

    n.stats()



    num_comps = 0
    for i in range(num_reviewers):
        print(i)

        if (i+1) % 1000 == 0:
            break

        # make queries unit length
        user = u[i] / np.linalg.norm(u[i])

        end = time.time()
        succes, p, stat_tracker = n.k_probe_approx(1, user, .1)
        end = time.time() - end
        #print(end)

        print('comparisons: ' + str(stat_tracker.get_stats()))

        comps, = stat_tracker.get_stats()

        num_comps += comps

        for v, idx in p:

            id = idx_to_id[idx]
            movie = id_to_movie[id]
            print(movie)

            #m_id = idx_to_id[idx]
            #movie_idx = int(ratings[idx][1])-1
            #print(idx, len(movies))
            #print(str(movies[idx]))

        #end = time.time()
        #true_max = n.find_max_inner(user)
        #end = time.time() - end
        #print(end)
        #print(movies[true_max[1]])
        #print(user.dot(true_max[0]))

    print(num_comps / 1000)

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
