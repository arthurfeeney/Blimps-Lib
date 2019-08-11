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
                    help='count of vectors in the dataset')
parser.add_argument('-d',
                    '--dim',
                    default=10,
                    metavar='N',
                    help='dimension of the vectors')
parser.add_argument('-q',
                    '--num_queries',
                    default=100,
                    metavar='N',
                    help='number of queries to test')
parser.add_argument('-a',
                    '--adj',
                    default=100,
                    metavar='N',
                    help='number of buckets to search in multiprobe')
args = parser.parse_args()


def main():
    global args

    # create the random dataset
    data = generate_data(args.count, args.dim)
    queries = generate_data(args.num_queries, args.dim)

    # create and fill the LSH table.
    t = nr_lsh.lsh_table(bits=32, dim=args.dim, num_buckets=args.count)
    t.fill(data, False)

    # "multiprobe" query the table.
    for query in queries:
        # find the nearset neighbor and its distance
        nn, dist = exact_neighbor(query, data)
        # the probe returns an option of KV, so it is MAYBE the
        # neighbor and its id; but, it could be None.
        maybe_neighbor, stats = t.probe(query, args.adj)
        approx_dist1 = np.linalg.norm(maybe_neighbor[0] - query)

        maybe_neighbor, stats = t.probe_approx(query, 2.0, args.adj)
        if (maybe_neighbor):
            approx_dist2 = np.linalg.norm(maybe_neighbor[0] - query)
        else:
            approx_dist2 = 0

        print('{0:.2f} {1:.2f} {2:.2f}'.format(approx_dist1, approx_dist2,
                                               dist))


def generate_data(count, dim):
    return np.random.randn(count, dim)


def exact_neighbor(query, data):
    m, mdist = None, 9999999
    for datum in data:
        dist = np.linalg.norm(query - datum)
        if dist < mdist:
            m, mdist = datum, dist
    return m, mdist


if __name__ == "__main__":
    main()
