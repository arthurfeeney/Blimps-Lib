

import nr_lsh as nr
import numpy as np
import matplotlib.pyplot as plt


for plt_num, dim in enumerate([2, 4, 8, 16, 32, 64]):

    bits = 32
    hash_max = 2**16-1

    hash = nr.simplelsh(bits, dim)

    query = np.random.randn(dim)
    query /= np.linalg.norm(query) # unit length
    q = hash.hash(query, hash_max)

    # generate a bunch of random vectors.
    data = (np.random.rand(100000, dim) * 2) - 1

    # normalize data to be l.e.q. to unit length and make things a bit more random
    for r in range(data.shape[0]):
        data[r] /= (np.linalg.norm(data[r]) + np.random.rand())

    inner_products = query.dot(data.transpose()).tolist()

    similarity = [0 for _ in range(data.shape[0])]
    for r in range(data.shape[0]):
        h1 = hash.hash(data[r], hash_max)
        similarity[r] = nr.same_bits(q, h1, 16)

    inner_products, similarity = zip(*sorted(zip(inner_products, similarity)))

    poly = np.polyfit(inner_products, similarity, 1)
    slope, inter = poly[0], poly[1]

    line = np.array(inner_products)*slope + inter


    plt.subplot(3, 2, plt_num+1)
    plt.plot(inner_products, similarity, color='red')
    plt.plot(inner_products, line, color='black')
    plt.title('dim=' + str(dim))
    plt.xlabel('inner product')
    plt.ylabel('similarity')


plt.tight_layout()
plt.show()
