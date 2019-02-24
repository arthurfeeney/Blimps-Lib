
import nr

class NormRangingTable(nr.MultiProbe):
    def __init__(self, num_partitions, bits, dim, num_buckets):
        super().__init__(num_partitions, bits, dim, num_buckets)
