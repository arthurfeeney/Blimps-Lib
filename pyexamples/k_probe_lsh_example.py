import argparse
import nr_lsh
import numpy as np

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
parser.add_argument('-b',
                    '--bits',
                    default=10,
                    metavar='N',
                    type=int,
                    help='the number of hash functions to use')
parser.add_argument('-q',
                    '--num_queries',
                    default=100,
                    metavar='N',
                    type=int,
                    help='number of queries to test')
parser.add_argument('-a',
                    '--adj',
                    default=100,
                    metavar='N',
                    type=int,
                    help='number of buckets to search in multiprobe')
parser.add_argument('-k', default=5, metavar='N', help="k to probe")
parser.add_argument('-v', default=False, metavar='S', help='verbose output')
args = parser.parse_args()


def main():
    global args

    # create the random dataset
    data = generate_data(args.count, args.dim)
    queries = generate_data(args.num_queries, args.dim)

    # create and fill the LSH table.
    t = nr_lsh.lsh_table(bits=args.bits, dim=args.dim, num_buckets=args.count)
    t.fill(data, False)

    # lists to track the recalls of each type of probe.
    k_probe_recall = []
    k_probe_approx_recall = []

    # "multiprobe" query the table.
    for query in queries:
        # find the true neighbors, indices in the dataset, and their distances
        indices, nns, dists = exact_k_neighbor(args.k, query, data)

        # the probe returns an option of KV, so it is MAYBE the
        # neighbor and its id; but, it could be None.
        maybe_neighbors, stats = t.k_probe(args.k, query, args.adj)
        neighbors = list(list(zip(*maybe_neighbors))[1])
        rec1 = recall(neighbors, indices)
        k_probe_recall.append(rec1)

        maybe_neighbors, stats = t.k_probe_approx(args.k, query, 2.0, args.adj)
        if (maybe_neighbors):
            neighbors = list(list(zip(*maybe_neighbors))[1])
            rec2 = recall(neighbors, indices)
        else:
            rec2 = 0
        k_probe_approx_recall.append(rec2)

        if args.v:
            print('{0:.2f} {1:.2f}'.format(rec1, rec2))

    print('mean recalls:')
    print(' * k_probe:        {0:.2f}'.format(
        sum(k_probe_recall) / len(k_probe_recall)))

    print(' * k_probe_approx: {0:.2f}'.format(
        sum(k_probe_approx_recall) / len(k_probe_approx_recall)))


def generate_data(count, dim):
    return np.random.randn(count, dim) * 5


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
