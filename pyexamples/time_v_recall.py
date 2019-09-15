import argparse
import nr_lsh
import numpy as np
import time
import matplotlib.pyplot as plt

#
# This example uses the LSH table to perform near neighbor search using
# random data.
#

parser = argparse.ArgumentParser(description='Parameters for LSH.')
parser.add_argument('-c',
                    '--count',
                    default=1000,
                    metavar='N',
                    type=int,
                    help='count of vectors in the dataset')
parser.add_argument('-d',
                    '--dim',
                    default=10,
                    metavar='N',
                    type=int,
                    help='dimension of the vectors')
parser.add_argument('-q',
                    '--num_queries',
                    default=100,
                    metavar='N',
                    type=int,
                    help='number of queries to test')
parser.add_argument('-a',
                    '--adj',
                    default=10,
                    metavar='N',
                    type=int,
                    help='number of buckets to search in multiprobe')
parser.add_argument('-k', default=5, metavar='N', help="k to probe")
args = parser.parse_args()


def main():
    global args

    # create the random dataset
    data = generate_data(args.count, args.dim)
    queries = generate_data(args.num_queries, args.dim)

    # create and fill the LSH table.
    mean_recalls = []
    mean_times = []

    for num_tables in range(5, 106, 20):
        #bits, num_tables = nr_lsh.sizes_from_probs(args.count, p1, p2)
        t = nr_lsh.lsh_table(num_tables=num_tables,
                             bits=32,
                             dim=args.dim,
                             num_buckets=args.count)
        t.fill(data, False)

        k_probe_recall = []
        time_per_query = []

        for query in queries:
            # find the true topk nearset neighbor and their distances
            # get indices of the true topk to compute recall.
            indices, nns, dists = exact_k_neighbor(args.k, query, data)

            # probe the table and record the time.
            # the probe returns an option of KV, so it is MAYBE the
            # neighbor and its id; but, it could be None.
            start = time.time()
            maybe_neighbors, stats = t.k_probe(args.k, query, args.adj)
            end = time.time()
            time_per_query.append(end - start)

            # compute revall if no neighbors were found, recall is 0
            if maybe_neighbors is None:
                rec1 = 0
            else:
                neighbors = list(list(zip(*maybe_neighbors))[1])
                rec1 = recall(neighbors, indices)
            k_probe_recall.append(rec1)

        mean_recalls.append(sum(k_probe_recall) / len(k_probe_recall))
        mean_times.append(sum(time_per_query) / len(time_per_query))

    plt.plot(mean_times, mean_recalls)
    plt.show()


def generate_data(count, dim):
    return np.random.randn(count, dim)


def exact_k_neighbor(k, query, data):
    # sort by distance and retrieve the indices of the k smallest
    dists_from_query = np.linalg.norm(data - query, axis=1)
    topk_indices = dists_from_query.argsort()[:k]
    topk = data[topk_indices]
    return topk_indices, topk, np.linalg.norm(topk - query)


def vect_in(vect, d):
    for row in d:
        if np.array_equal(vect, row):
            return True
    return False


def generate_data(count, dim):
    return np.random.randn(count, dim)


def exact_k_neighbor(k, query, data):
    # sort by distance and retrieve the indices of the k smallest
    dists_from_query = np.linalg.norm(data - query, axis=1)
    topk_indices = dists_from_query.argsort()[:k]
    topk = data[topk_indices]
    return topk_indices, topk, np.linalg.norm(topk - query)


def vect_in(vect, d):
    for row in d:
        if np.array_equal(vect, row):
            return True
    return False


def recall(d1, d2):
    # takes two lists of indices
    # finds the number of overlapping elements.
    return len([a for a in d1 if a in d2]) / len(d2)


if __name__ == "__main__":
    main()
