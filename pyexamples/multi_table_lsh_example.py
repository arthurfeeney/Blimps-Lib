import argparse
import nr_lsh
import numpy as np
import time

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
args = parser.parse_args()


def main():
    global args

    # create the random dataset
    data = generate_data(args.count, args.dim)
    queries = generate_data(args.num_queries, args.dim)

    # create and fill the LSH table.
    t = nr_lsh.lsh_table(num_tables=20,
                         bits=32,
                         dim=args.dim,
                         num_buckets=args.count)
    t.fill(data, False)

    # track the total time to perform all queries.
    exact_total_time = 0
    probe_total_time = 0
    probe_approx_total_time = 0

    # "multiprobe" query the table.
    for query in queries:
        # find the nearset neighbor and its distance
        start = time.time()
        nn, dist = exact_neighbor(query, data)
        end = time.time()
        exact_total_time += (end - start)
        # probe the table and record the time.
        start = time.time()
        # the probe returns an option of KV, so it is MAYBE the
        # neighbor and its id; but, it could be None.
        maybe_neighbor, stats = t.probe(query, args.adj)
        end = time.time()
        probe_total_time += (end - start)

        approx_dist1 = np.linalg.norm(maybe_neighbor[0] - query)

        start = time.time()
        maybe_neighbor, stats = t.probe_approx(query, 2.0, args.adj)
        end = time.time()
        probe_approx_total_time += (end - start)
        if (maybe_neighbor):
            approx_dist2 = np.linalg.norm(maybe_neighbor[0] - query)
        else:
            approx_dist2 = 0

        print('{0:.2f} {1:.2f} {2:.2f}'.format(approx_dist1, approx_dist2,
                                               dist))
    print('Exact Average Time Per Query: ' +
          str(exact_total_time / args.num_queries))
    print('Probe Average Time Per Query: ' +
          str(probe_total_time / args.num_queries))
    print('Probe Approx Average Time Per Query: ' +
          str(probe_approx_total_time / args.num_queries))


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
