

import nr
import numpy as np
import scipy.sparse

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def main():
    num_movies = 65133
    num_reviewers = 71567 

    ratings_file = open('ml-10m/ml-10M100K/ratings.dat', 'r+')

    ratings_line = ratings_file.readlines()

    ratings = [line.split('::') for line in ratings_line]
    ratings = [tuple(map(float, r[:-1])) for r in ratings]

    data = [r[2] for r in ratings]
    user_id = [int(r[0])-1 for r in ratings]
    movie_id = [int(r[1])-1 for r in ratings]

    review_matrix_csr = csr_matrix((data, (user_id, movie_id)), 
                                   shape=(num_reviewers, num_movies))

    u, s, vt = svds(review_matrix_csr, k=150)

    n = nr.MultiTable(30, 32, 16, 150)

    n.fill(vt.transpose())

    user = u[1]

    success, (q, index) = n.MIPS(user)

    print(q)

    movies_file = open('ml-10m/ml-10M100K/movies.dat', 'r+')
    movies_line = movies_file.readlines()
    movies = [line.split('::') for line in movies_line]

    print(movies[index])
    print(q.dot(user))


if __name__ == "__main__":
    main()
