import numpy as np

#
# Computes precision and recall using method discussed in PureSVD paper.
# by Cremonesi et al. It basically checks if movies highly rated by the user
# are put in the top k of N additional, unrated movies.
#

def find_five_rating(user_ratings, mean_rating):
    # return the indices of all 5-star ratings.
    # user_ratings should be a list.
    indices = [i for i in range(user_ratings.size) if user_ratings[i] == 5 - mean_rating]
    return indices

def get_k_unrated(k, user_ratings):
    # randomly select k items (i.e., 1000 used by pureSVD) that are
    # unrated by the user.
    unrated_indices = [idx for idx in range(user_ratings.size) if user_ratings[idx] == 0]
    return np.random.choice(unrated_indices,
                            size=k,
                            replace=False) # won't select the same index many times

def random_item_ranked_list(k, user_ratings, five_star_idx, user_factors, item_factors):
    unrated_indices = get_k_unrated(k, user_ratings)
    unrated_item_factors = item_factors[:,unrated_indices] # column matrix
    #ratings = user_factors.dot(unrated_item_factors)
    ratings = user_ratings.dot(item_factors.transpose()).dot(unrated_item_factors)
    five_star_rating = user_ratings.dot(item_factors.transpose()).dot(item_factors[:,five_star_idx])
    #ratings = user_factors.dot(unrated_item_factors)
    #five_star_rating = user_factors.dot(item_factors[:,five_star_idx])
    ratings.sort()
    print(ratings)
    return five_star_rating, ratings

def hit(N, five_star_rating, ratings):
    # if five_star_rating is larger than the smallest item in the top N,
    # it is a hit
    # if N=1, we want to check that the five_star_rating larger than the
    # highest rating.
    return five_star_rating > ratings[-N]

def user_recall(k, N, user_ratings, user_factors, item_factors):
    # computes the precision for a single user.
    # checks if the system puts items rated 5 into the top k of N + 1 items.
    five_star_indices = find_five_rating(user_ratings)

    # if a user never gives something five stars, returning 1 effectively
    # "ignores" them.
    if(len(five_star_indices) == 0):
        return 0

    total_hits = 0
    for five_star_idx in five_star_indices:
        fsr, ratings = random_item_ranked_list(k, user_ratings, five_star_idx,
                                               user_factors, item_factors)
        total_hits += hit(N, fsr, ratings)
    return total_hits / len(five_star_indices)

def user_precision(k, N, user_ratings, user_factors, item_factors):
    return user_recall(k, N, user_ratings, user_factors, item_factors) / N

def recall(k, N, test_ratings, item_factors, review_matrix_csr, mean_rating):
    total_hits = 0
    for (useridx, movieidx, rating) in test_ratings:
        assert rating == 5 - mean_rating # test ratings should only be 5.
        user_ratings = review_matrix_csr[useridx].toarray()[0]
        unrated_indices = get_k_unrated(k, user_ratings)
        unrated_item_factors = item_factors[unrated_indices] # column matrix
        ratings = user_ratings.dot(item_factors).dot(unrated_item_factors.transpose())
        five_star_rating = user_ratings\
            .dot(item_factors)\
            .dot(item_factors[int(movieidx)])
        ratings.sort()
        total_hits += hit(N, five_star_rating, ratings)
    return total_hits / len(test_ratings)

def precision(k, N, test_ratings, item_factors, review_matrix_csr, mean_rating):
    return recall(k, N, test_ratings, item_factors, review_matrix_csr, mean_rating) / N
