
import nr
import numpy as np

lsh = nr.MultiTable(2, 4, 16, 30)


#data = np.array([[123,234,123,34],[1,234,234,234]])

data = [np.random.randn(30) for _ in range(1000000)]

lsh.fill(data)

query = np.random.randn(30)

query /= np.linalg.norm(query)

m = lsh.MIPS(query)
t = m[1][0]

print(query)
print(m)

print(t.dot(query))

