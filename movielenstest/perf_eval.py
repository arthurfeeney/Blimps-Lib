import numpy as np

#
# Computes precision and recall using method discussed in PureSVD paper.
# by Cremonesi et al. It basically checks if movies highly rated by the user
# are put in the top k of N additional, unrated movies.
#


def find_five_rating(user_ratings, mean_rating):
    # return the indices of all 5-star ratings.
    # user_ratings should be a list.
    indices = [
        i for i in range(user_ratings.size)
        if user_ratings[i] == 5 - mean_rating
    ]
    return indices


def get_k_unrated(k, user_ratings):
    # randomly select k items (i.e., 1000 used by pureSVD) that are
    # unrated by the user.
    unrated_indices = [
        idx for idx in range(user_ratings.size) if user_ratings[idx] == 0
    ]
    return np.random.choice(
        unrated_indices, size=k,
        replace=False)  # won't select the same index many times


def random_item_ranked_list(k, user_ratings, five_star_idx, user_factors,
                            item_factors):
    unrated_indices = get_k_unrated(k, user_ratings)
    unrated_item_factors = item_factors[:, unrated_indices]  # column matrix
    #ratings = user_factors.dot(unrated_item_factors)
    ratings = user_ratings.dot(
        item_factors.transpose()).dot(unrated_item_factors)
    five_star_rating = user_ratings.dot(item_factors.transpose()).dot(
        item_factors[:, five_star_idx])
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
    if isinstance(N, int):
        return five_star_rating > ratings[-N]
    else:
        return [five_star_rating > ratings[-n] for n in N]


#
# Precision and Recall for the exact case:
#


def user_recall(k, N, user_ratings, user_factors, item_factors, mean_rating):
    # computes the precision for a single user.
    # checks if the system puts items rated 5 into the top k of N + 1 items.
    five_star_indices = find_five_rating(user_ratings, mean_rating)

    # if a user never gives something five stars, returning 1 effectively
    # "ignores" them.
    if (len(five_star_indices) == 0):
        return 0

    total_hits = 0
    for five_star_idx in five_star_indices:
        fsr, ratings = random_item_ranked_list(k, user_ratings, five_star_idx,
                                               user_factors, item_factors)
        total_hits += hit(N, fsr, ratings)
    return total_hits / len(five_star_indices)


def user_precision(k, N, user_ratings, user_factors, item_factors,
                   mean_rating):
    return user_recall(k, N, user_ratings, user_factors, item_factors,
                       mean_rating) / N


def recall(k, N, test_ratings, item_factors, review_matrix_csr, mean_rating):

    # when N is an integral type, make it a list so it is iterable.
    if isinstance(N, int):
        N = [N]

    total_hits = dict([(n, 0) for n in N])
    for (useridx, movieidx, rating) in test_ratings:
        assert rating == 5 - mean_rating  # all test ratings are 5 - mean
        user_ratings = review_matrix_csr[useridx].toarray()[0]
        unrated_indices = get_k_unrated(k, user_ratings)
        unrated_item_factors = item_factors[unrated_indices]
        ratings = user_ratings.dot(item_factors).dot(
            unrated_item_factors.transpose())
        five_star_rating = user_ratings\
            .dot(item_factors)\
            .dot(item_factors[int(movieidx)])
        ratings.sort()

        is_hit = hit(N, five_star_rating, ratings)
        for (idx, n) in enumerate(N):
            total_hits[n] += is_hit[idx]

    for n in N:
        total_hits[n] /= len(test_ratings)
    return total_hits


def precision(k, N, test_ratings, item_factors, review_matrix_csr,
              mean_rating):
    if isinstance(N, int):
        # make N iterable if it is an int.
        N = [N]
    rec = recall(k, N, test_ratings, item_factors, review_matrix_csr,
                 mean_rating)
    for key in rec:
        rec[key] /= key
    return rec


def recall_and_precision(k, N, test_ratings, item_factors, review_matrix_csr,
                         mean_rating):
    # dont have to compute recall and precision separately. This avoids some
    # unnecessary computations.
    if isinstance(N, int):
        # make N iterable if it is an int.
        N = [N]
    rec = recall(k, N, test_ratings, item_factors, review_matrix_csr,
                 mean_rating)
    prec = dict([(key, rec[key] / key) for key in rec])
    return rec, prec


#
# Precision and Recall for MIPS.
# Uses method described in "on Sym and Asym LSH paper."
#


def topk_inner(k, inner):
    # compute the topk largest inner products.
    indices = (-inner).argsort()[:k]
    return inner[indices], indices


def fraction_intersect(l1, l2):
    # computes the fraction of elements in l2 that are also in l1.
    return len([a for a in l2 if a in l1]) / len(l1)


def MIPS_recall(k, test_ratings, item_factors, nr_table, review_matrix_csr,
                mean_rating):
    # computes the average overlap of topk ratings on the test set.
    total_recall = 0
    total_hits = 0
    ctr = 0
    for (useridx, movieidx, rating) in test_ratings:
        ctr += 1
        print(ctr)

        assert rating == 5 - mean_rating
        user_ratings = review_matrix_csr[useridx].toarray()[0]

        # get the true topk.
        true_topk, true_idx = topk_inner(
            k,
            user_ratings.dot(item_factors).dot(item_factors.transpose()))

        # probe k from the nr table.
        query = user_ratings.dot(item_factors)
        query /= np.linalg.norm(query)  # queries are unit vectors
        data, tracker = nr_table.k_probe(k, query, int(5))

        if data:
            _, approx_idx = zip(*data)
            # find the fraction of k_probe that are truly in the topk.
            total_recall += fraction_intersect(true_idx, approx_idx)
            # check if is 5 star movie in the topk.
            total_hits += movieidx in approx_idx
    # average % recall across test.
    #return total_recall / len(test_ratings), total_hits / len(test_ratings)
    return total_recall / ctr, total_hits / ctr
