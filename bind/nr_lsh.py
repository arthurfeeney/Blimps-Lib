#
# This defines the 'real' package to include.
# includes better helper stuff than the bindings.
#

import numpy as np
import nr_binding


class float32(np.float32):
    pass


class float64(np.float64):
    pass


_valid_float32 = [float32, np.float32, 'float32']
_valid_float64 = [float64, np.float64, 'float64']


def multiprobe(num_tables,
               num_partitions,
               bits,
               dim,
               num_buckets,
               dtype=float32):
    if any([dtype is f for f in _valid_float32]):
        return nr_binding.MultiProbeFloat(num_tables, num_partitions, bits,
                                          dim, num_buckets)
    elif any([dtype is f for f in _valid_float64]):
        return nr_binding.MultiProbeDouble(num_tables, num_partitions, bits,
                                           dim, num_buckets)
    else:
        raise ValueError('multiprobe', 'invalid dtype', 'dtype=' + str(dtype))
