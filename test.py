import numpy as np
from math import nan

x = np.array([1, 2, 3, nan, nan])
print((~np.isnan(x)).sum())