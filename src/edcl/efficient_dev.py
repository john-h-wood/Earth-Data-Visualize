import numpy as np

a = float('nan')

test_list = np.array([a, 1, 2, 3, a, 4, 5, a])
print(test_list)
test_list.sort()
print(test_list)

print(np.searchsorted(test_list, 100))



