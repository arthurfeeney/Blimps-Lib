import numpy as np

#
# Computes precision and recall using method discussed in PureSVD paper.
# by Cremonesi et al. It basically checks if movies highly rated by the user
# are put in the top k of N additional, unrated movies.
#

def find_five_rating(user_ratings):
    # return the indices of all 5-star ratings.
    # user_ratings should be a list.
    indices = [i for i in range(user_ratings.size) if user_ratings[i] == 5]
    return indices

def get_k_unrated(k, user_ratings):
    # randomly select k items (i.e., 1000 used by pureSVD) that are
    # unrated by the user.
    #enum_ratings = np.ndenumerate(user_ratings)
    unrated_indices = [idx for idx in range(user_ratings.size) if user_ratings[idx] == 0]
    #unrated_indices = [idx[0] for (idx, r) in enum_ratings if r == 0]
    return np.random.choice(unrated_indices,
                            size=k,
                            replace=False)

def random_item_ranked_list(k, user_ratings, five_star_idx, user_factors, item_factors):
    unrated_indices = get_k_unrated(k, user_ratings)
    unrated_item_factors = item_factors[:,unrated_indices] # column matrix
    ratings = user_factors.dot(unrated_item_factors)
    five_star_rating = user_factors.dot(item_factors[:,five_star_idx])
    ratings.sort()
    return five_star_rating, ratings

def hit(N, five_star_rating, ratings):
    # if five_star_rating is in the top N ratings, we have a hit.
    # ratings is in ascending order. So, they must be reversed.
    for rating in ratings[::-1][:N]: # iterate though n largest.
        if five_star_rating > rating:
            return True
    return False

def user_recall(k, N, user_ratings, user_factors, item_factors):
    # computes the precision for a single user.
    # checks if the system puts items rated 5 into the top k of N + 1 items.
    five_star_indices = find_five_rating(user_ratings)

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
