import numpy as np

def find_five_rating(user, rating_matrix):
    # return the index of all 5-star ratings.
    ratings = rating_matrix[user] # ratings is an array
    indices = [i for i in range(ratings.size) if ratings[i] == 5]
    return indices

def get_k_unrated(k, user, rating_matrix):
    # randomly select k items (i.e., 1000 used by pureSVD) that are
    # unrated by the user.
    user_ratings = rating_matrix[user]
    enum_ratings = np.ndenumerate(user_ratings)
    unrated_indices = [idx[0] for (idx, r) in enum_ratings if r == 0]
    return np.random.choice(unrated_indices,
                            size=k,
                            replace=False)

def random_item_ranked_list(k, user, rating_matrix, user_factors, item_factors):
    unrated_indices = get_k_unrated(k, user, rating_matrix)
    #user_vect = user_factors[user] # each user vector is a row.
    #unrated_item_factors = item_factors[:,unrated_indices] # column major
    #ratings = user_vect.dot(unrated_item_factors)
